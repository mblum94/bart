/* Copyright 2022. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <stdio.h>
#include <memory.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/linalg.h"
#include "num/init.h"
#include "num/ode.h"

#include "simu/bloch.h"
#include "simu/pulse.h"

#include "simulation.h"


void debug_sim(struct sim_data* data)
{
        debug_printf(DP_WARN, "Simulation-Debug-Output\n\n");
        debug_printf(DP_WARN, "Voxel-Parameter:\n");
        debug_printf(DP_INFO, "\tR1:%f\n", data->voxel.r1);
        debug_printf(DP_INFO, "\tR2:%f\n", data->voxel.r2);
        debug_printf(DP_INFO, "\tM0:%f\n", data->voxel.m0);
        debug_printf(DP_INFO, "\tw:%f\n", data->voxel.w);
        debug_printf(DP_INFO, "\tB1:%f\n\n", data->voxel.b1);

        debug_printf(DP_WARN, "Seq-Parameter:\n");
        debug_printf(DP_INFO, "\tSimulation Type:%d\n", data->seq.type);
        debug_printf(DP_INFO, "\tSequence:%d\n", data->seq.seq_type);
        debug_printf(DP_INFO, "\tTR:%f\n", data->seq.tr);
        debug_printf(DP_INFO, "\tTE:%f\n", data->seq.te);
        debug_printf(DP_INFO, "\t#Rep:%d\n", data->seq.rep_num);
        debug_printf(DP_INFO, "\t#Spins:%d\n", data->seq.spin_num);
        debug_printf(DP_INFO, "\tIPL:%f\n", data->seq.inversion_pulse_length);
        debug_printf(DP_INFO, "\tPPL:%f\n", data->seq.prep_pulse_length);
        debug_printf(DP_INFO, "\tPulse Applied?:%d\n\n", data->seq.pulse_applied);

        debug_printf(DP_WARN, "Gradient-Parameter:\n");
        debug_printf(DP_INFO, "\tMoment:%f\n", data->grad.mom);
        debug_printf(DP_INFO, "\tMoment SL:%f\n\n", data->grad.mom_sl);

        debug_printf(DP_WARN, "Pulse-Parameter:\n");
        debug_printf(DP_INFO, "\tRF-Start:%f\n", data->pulse.rf_start);
        debug_printf(DP_INFO, "\tRF-End:%f\n", data->pulse.rf_end);
        debug_printf(DP_INFO, "\tFlipangle:%f\n", data->pulse.flipangle);
        debug_printf(DP_INFO, "\tPhase:%f\n", data->pulse.phase);
        debug_printf(DP_INFO, "\tBWTP:%f\n", data->pulse.bwtp);
        debug_printf(DP_INFO, "\tNL:%f\n", data->pulse.nl);
        debug_printf(DP_INFO, "\tNR:%f\n", data->pulse.nr);
        debug_printf(DP_INFO, "\tN:%f\n", data->pulse.n);
        debug_printf(DP_INFO, "\tt0:%f\n", data->pulse.t0);
        debug_printf(DP_INFO, "\tAlpha:%f\n", data->pulse.alpha);
        debug_printf(DP_INFO, "\tA:%f\n\n", data->pulse.A);

        debug_printf(DP_WARN, "Inversion Pulse-Parameter:\n");
        debug_printf(DP_INFO, "\tA0:%f\n", data->pulse.hs.a0);
        debug_printf(DP_INFO, "\tBeta:%f\n", data->pulse.hs.beta);
        debug_printf(DP_INFO, "\tMu:%f\n", data->pulse.hs.mu);
        debug_printf(DP_INFO, "\tDuration:%f\n", data->pulse.hs.duration);
        debug_printf(DP_INFO, "\tON?:%d\n", data->pulse.hs.on);
}


const struct simdata_voxel simdata_voxel_defaults = {

	.r1 = 0.,
	.r2 = 0.,
	.m0 = 1.,
	.w = 0.,
	.b1 = 1.,
};


const struct simdata_seq simdata_seq_defaults = {

        .type = SIM_ODE,
	.seq_type = SEQ_BSSFP,
	.tr = 0.004,
	.te = 0.002,
	.rep_num = 1,
	.spin_num = 1,

        .perfect_inversion = false,
	.inversion_pulse_length = 0.01,
        .inversion_spoiler = 0.,

	.prep_pulse_length = 0.001,

        .pulse_applied = false,
};


const struct simdata_tmp simdata_tmp_defaults = {

        .rep_counter = 0,
	.spin_counter = 0,
	.t = 0.,
	.w1 = 0.,
	.r2spoil = 0.,
};


const struct simdata_grad simdata_grad_defaults = {

	.gb = { 0., 0., 0. },
	.gb_eff = { 0., 0., 0.},
	.mom = 0.,
	.mom_sl = 0.,
};


/* --------- Matrix Operations --------- */


static void vm_mul_transpose(int N, float out[N], float matrix[N][N], float in[N])
{
	for (int i = 0; i < N; i++) {

		out[i] = 0.;

		for (int j = 0; j < N; j++)
			out[i] += matrix[j][i] * in[j];
	}
}


static void mm_mul(int N, float out[N][N], float in1[N][N], float in2[N][N])
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {

			out[i][j] = 0.;

			for (int k = 0; k < N; k++)
				out[i][j] += in1[i][k] * in2[k][j];
		}
}


/* ------------ Bloch Equations -------------- */

static void set_gradients(void* _data, float t)
{
        struct sim_data* data = _data;

	if (data->seq.pulse_applied) {

                // Hyperbolic Secant pulse
                if (data->pulse.hs.on) {

                        data->tmp.w1 = pulse_hypsec_am(&data->pulse.hs, t);

                        data->pulse.phase = pulse_hypsec_phase(&data->pulse.hs, t);

                } else { //Windowed Sinc pulse

                        data->tmp.w1 = pulse_sinc(&data->pulse, t);
                }

		data->grad.gb_eff[0] = cosf(data->pulse.phase) * data->tmp.w1 * data->voxel.b1 + data->grad.gb[0];
		data->grad.gb_eff[1] = sinf(data->pulse.phase) * data->tmp.w1 * data->voxel.b1 + data->grad.gb[1];

	} else {

		data->tmp.w1 = 0.;
		data->grad.gb_eff[0] = data->grad.gb[0];
		data->grad.gb_eff[1] = data->grad.gb[1];
	}

	// Units: [gb] = rad/s
	data->grad.gb_eff[2] = data->grad.gb[2];
}


/* --------- ODE Simulation --------- */

static void bloch_simu_ode_fun(void* _data, float* out, float t, const float* in)
{
        struct sim_data* data = _data;

        set_gradients(data, t);

	bloch_ode(out, in, data->voxel.r1, data->voxel.r2+data->tmp.r2spoil, data->grad.gb_eff);
}


static void bloch_pdy2(void* _data, float* out, float t, const float* in)
{
	struct sim_data* data = _data;
	(void)t;

	bloch_pdy((float(*)[3])out, in, data->voxel.r1, data->voxel.r2 + data->tmp.r2spoil, data->grad.gb_eff);
}


static void bloch_pdp2(void* _data, float* out, float t, const float* in)
{
	struct sim_data* data = _data;
	(void)t;

	bloch_b1_pdp((float(*)[3])out, in, data->voxel.r1, data->voxel.r2 + data->tmp.r2spoil, data->grad.gb_eff, data->pulse.phase, data->tmp.w1);
}


/* ---------  State-Transition Matrix Simulation --------- */


static void bloch_simu_stm_fun(void* _data, float* out, float t, const float* in)
{
        struct ode_matrix_simu_s* ode_data = _data;
	struct sim_data* data = ode_data->sim_data;

        unsigned int N = ode_data->N;

        set_gradients(data, t);

	float matrix_time[N][N];

	bloch_matrix_ode_sa2(matrix_time, data->voxel.r1, data->voxel.r2+data->tmp.r2spoil, data->grad.gb_eff, data->pulse.phase, data->tmp.w1);

        for (unsigned int i = 0; i < N; i++) {

		out[i] = 0.;

		for (unsigned int j = 0; j < N; j++)
			out[i] += matrix_time[i][j] * in[j];
	}
}


void ode_matrix_interval_simu(struct sim_data* _data, float h, float tol, unsigned int N, float out[N], float st, float end)
{
        struct ode_matrix_simu_s data = { N, _data };
	ode_interval(h, tol, N, out, st, end, &data, bloch_simu_stm_fun);
}


void mat_exp_simu(struct sim_data* data, int N, float st, float end, float out[N][N])
{
	assert(end >= st);

	// compute F(t) := exp(tA)
	// F(0) = id
	// d/dt F = A

	float h = (end-st) / 100.;
	float tol = 1.E-6;

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++)
			out[i][j] = (i == j) ? 1. : 0.;

		ode_matrix_interval_simu(data, h, tol, N, out[i], st, end);
	}
}

static void create_sim_matrix(struct sim_data* data, int N, float matrix[N][N], float st, float end)
{
	if (data->seq.pulse_applied)
		sinc_pulse_create(&data->pulse, data->pulse.rf_start, data->pulse.rf_end, data->pulse.flipangle, data->pulse.phase, data->pulse.bwtp, data->pulse.alpha);

	mat_exp_simu(data, N, st, end, matrix);
}

static void apply_sim_matrix(int N, float m[N], float matrix[N][N])
{
	float tmp[N];

	for (int i = 0; i < N; i++)
		tmp[i] = m[i];

	vm_mul_transpose(N, m, matrix, tmp);
}


/* ------------ Read-Out -------------- */

static void adc_corr(int N, int P, float out[P][N], float in[P][N], float angle)
{
	for (int i = 0; i < P; i++)
		rotz(out[i], in[i], angle);
}


static long vector_position(int d, int r, int rep_max, int s, int spin_max)
{
        return d * spin_max * rep_max + r * spin_max + s;
}


static void collect_signal(struct sim_data* data, int N, int P, float* mxy, float* sa_r1, float* sa_r2, float* sa_b1, float xp[P][N])
{
	float tmp[4][3] = { { 0. }, { 0. }, { 0. }, { 0. } };

	adc_corr(N, P, tmp, xp, -data->pulse.phase);

        long ind = 0;

	for (int i = 0; i < N; i++) {

                ind = vector_position(i, data->tmp.rep_counter, data->seq.rep_num, data->tmp.spin_counter, data->seq.spin_num);

		if (NULL != mxy)
			mxy[ind] = tmp[0][i];

		if (NULL != sa_r1)
			sa_r1[ind] = tmp[1][i];

		if (NULL != sa_r2)
			sa_r2[ind] = tmp[2][i];

		if (NULL != sa_b1)
			sa_b1[ind] = tmp[3][i];
	}
}


static void sum_up_signal(struct sim_data* data, float *mxy,  float *sa_r1, float *sa_r2, float *sa_b1,
                        float (*mxy_sig)[3], float (*sa_r1_sig)[3], float (*sa_r2_sig)[3], float (*sa_m0_sig)[3], float (*sa_b1_sig)[3])
{
        float sum_mxy;
	float sum_sa_r1;
	float sum_sa_r2;
	float sum_sa_b1;

        long ind = 0;

        // Dimensions; [x, y, z]
	for (int dim = 0; dim < 3; dim++) {

		sum_mxy = 0.;
		sum_sa_r1 = 0.;
		sum_sa_r2 = 0.;
		sum_sa_b1 = 0.;

                //Repetitions
		for (int r = 0; r < data->seq.rep_num; r++) {

                        // Spins
			for (int spin = 0; spin < data->seq.spin_num; spin++) {

                                ind = vector_position(dim, r, data->seq.rep_num, spin, data->seq.spin_num);

				sum_mxy += mxy[ind];
				sum_sa_r1 += sa_r1[ind];
				sum_sa_r2 += sa_r2[ind];
				sum_sa_b1 += sa_b1[ind];
			}

                        // Mean
                        mxy_sig[r][dim] = sum_mxy * data->voxel.m0 / (float)data->seq.spin_num;
                        sa_r1_sig[r][dim] = sum_sa_r1 * data->voxel.m0 / (float)data->seq.spin_num;
                        sa_r2_sig[r][dim] = sum_sa_r2 * data->voxel.m0 / (float)data->seq.spin_num;
                        sa_b1_sig[r][dim] = sum_sa_b1 * data->voxel.m0 / (float)data->seq.spin_num;
                        sa_m0_sig[r][dim] = sum_mxy / (float)data->seq.spin_num;

                        sum_mxy = 0.;
                        sum_sa_r1 = 0.;
                        sum_sa_r2 = 0.;
                        sum_sa_b1 = 0.;
		}
	}
}

/* ------------ RF-Pulse -------------- */

// Single hard pulse without discrete sampling
static void hard_pulse(struct sim_data* data, int N, int P, float xp[P][N])
{
        for (int i = 0; i < P; i++)
                bloch_excitation2(xp[i], xp[i], data->pulse.flipangle / 180. * M_PI, data->pulse.phase);
}


void start_rf_pulse(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float stm_matrix[P * N][P * N])
{
	data->seq.pulse_applied = true;

        // Define effective z Gradient = Slice-selection gradient + off-resonance [rad/s]
	data->grad.gb[2] = data->grad.mom_sl + data->voxel.w;

        switch (data->seq.type) {

        case SIM_ODE:
                ;
                if (0. == data->pulse.rf_end)
                        hard_pulse(data, N, P, xp);
                else
                        // Choose P-1 because ODE interface treats signal seperat and P only describes the number of parameters
	                ode_direct_sa(h, tol, N, P - 1, xp, data->pulse.rf_start, data->pulse.rf_end, data,  bloch_simu_ode_fun, bloch_pdy2, bloch_pdp2);
                break;

        case SIM_STM:
                ;
                create_sim_matrix(data, P * N, stm_matrix, data->pulse.rf_start, data->pulse.rf_end);
                break;
        }

        data->grad.gb[2] = 0.;
}


/* ------------ Relaxation -------------- */

static void hard_relaxation(struct sim_data* data, int N, int P, float xp[P][N], float st, float end)
{
	float xp2[3] = { 0. };

	for (int i = 0; i < P; i++) {

		xp2[0] = xp[i][0];
		xp2[1] = xp[i][1];
		xp2[2] = xp[i][2];

		bloch_relaxation(xp[i], end-st, xp2, data->voxel.r1, data->voxel.r2+data->tmp.r2spoil, data->grad.gb);
	}
}


static void relaxation2(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end, float stm_matrix[P * N][P * N])
{
	data->seq.pulse_applied = false;

        // Define effective z Gradient =Gradient Moments + off-resonance [rad/s]
        data->grad.gb[2] = data->grad.mom + data->voxel.w;

        switch (data->seq.type) {

        case SIM_ODE:
                ;
                if (0. == data->pulse.rf_end)
                        hard_relaxation(data, N, P, xp, st, end);
                else
                        // Choose P-1 because ODE interface treats signal seperat and P only describes the number of parameters
	                ode_direct_sa(h, tol, N, P - 1, xp, st, end, data, bloch_simu_ode_fun, bloch_pdy2, bloch_pdp2);
                break;

        case SIM_STM:
                ;
                create_sim_matrix(data, P * N, stm_matrix, st, end);
                break;
        }

        data->grad.gb[2] = 0.;
}


/* ------------ Conversion ODE -> STM -------------- */


static void stm2ode(int N, int P, float out[P][N], float in[P * N + 1])
{
        for (int p = 0; p < P; p++)
                for(int n = 0; n < N; n++)
                        out[p][n] = in[p * N + n];
}

static void ode2stm(int N, int P, float out[P * N + 1], float in[P][N])
{
        for (int p = 0; p < P; p++)
                for(int n = 0; n < N; n++)
                        out[p * N + n] = in[p][n];

        out[P * N] = 1.;
}


/* ------------ Structural Elements -------------- */

static void prepare_sim(struct sim_data* data, int N, int P, float mte[P * N + 1][P * N + 1], float mtr[P * N + 1][P * N + 1])
{
        switch (data->seq.type) {

        case SIM_ODE:
                ;
                if (0. != data->pulse.rf_end)
                	sinc_pulse_create(&data->pulse, data->pulse.rf_start, data->pulse.rf_end, data->pulse.flipangle, data->pulse.phase, data->pulse.bwtp, data->pulse.alpha);

                break;

        case SIM_STM:
                ;
                int M = P*N+1;

                // Matrix: 0 -> T_RF
                float mrf[M][M];
                start_rf_pulse(data, 0., 0., M, 1, NULL, mrf);

                // Matrix: T_RF -> TE
                float mrel[M][M];

                float tmp[M][M];
                float tmp2[M][M];

                if (0 != data->grad.mom_sl) {

                        if (0.0000001 > (1.5*data->pulse.rf_end - data->seq.te)) { // Catch equality of floats

                                // Slice-Rewinder

                                data->grad.mom = -data->grad.mom_sl;
                                relaxation2(data, 0, 0, M, 1, NULL, data->pulse.rf_end, 1.5 * data->pulse.rf_end, tmp);
                                data->grad.mom = 0.; // [rad/s]

                                relaxation2(data, 0, 0, M, 1, NULL, 1.5 * data->pulse.rf_end, data->seq.te, tmp2);

                                mm_mul(M, mrel, tmp, tmp2);

                        } else
                                debug_printf(DP_WARN, "Slice-Selection Gradient rewinder does not fit between RF_end and TE!\n");
                } else {

                        relaxation2(data, 0., 0., M, 1, NULL, data->pulse.rf_end, data->seq.te, mrel);
                }

                // Join matrices: 0 -> TE
                mm_mul(M, mte, mrf, mrel);

                // Smooth spoiling for FLASH sequences

                if (    (SEQ_FLASH == data->seq.seq_type) ||
                        (SEQ_IRFLASH == data->seq.seq_type))

		        data->tmp.r2spoil = 10000.;

                // Balance z-gradient for bSSFP type sequences

                if (    (SEQ_BSSFP == data->seq.seq_type) ||
                        (SEQ_IRBSSFP == data->seq.seq_type)) {

                        // Matrix: TE -> TR-T_RF
                        relaxation2(data, 0., 0., M, 1, NULL, data->seq.te, data->seq.tr-data->pulse.rf_end, tmp);

                        // Matrix: TR-T_RF -> TR
                        data->grad.mom = -data->grad.mom_sl;
                        relaxation2(data, 0., 0., M, 1, NULL, data->seq.tr-data->pulse.rf_end, data->seq.tr, tmp2);
                        data->grad.mom = 0.;

                        // Join matrices: TE -> TR
                        mm_mul(M, mtr, tmp, tmp2);

                } else {

                        relaxation2(data, 0., 0., M, 1, NULL, data->seq.te, data->seq.tr, mtr);
                }

                data->tmp.r2spoil = 0.;	// effects spoiled sequences only

                break;
        }

}


static void run_sim(struct sim_data* data, float* mxy, float* sa_r1, float* sa_r2, float* sa_b1,
                        float h, float tol, int N, int P, float xp[P][N],
                        float xstm[P * N + 1], float mte[P * N + 1][P * N + 1], float mtr[P * N + 1][P * N + 1],
                        bool get_signal)
{
        switch (data->seq.type) {

        case SIM_ODE:
                ;
                start_rf_pulse(data, h, tol, N, P, xp, NULL);

                // Slice-Rewinder if time is long enough

                if (0 != data->grad.mom_sl) {

                        if (0.0000001 > (1.5 * data->pulse.rf_end - data->seq.te)) { // Catch also equality of floats

                                data->grad.mom = -data->grad.mom_sl;
                                relaxation2(data, h, tol, N, P, xp, data->pulse.rf_end, 1.5 * data->pulse.rf_end, NULL);
                                data->grad.mom = 0.; // [rad/s]

                                relaxation2(data, h, tol, N, P, xp, 1.5 * data->pulse.rf_end, data->seq.te, NULL);

                        } else {

                                debug_printf(DP_WARN, "Slice-Selection Gradient rewinder does not fit between RF_end and TE!\n");
                        }

                } else {
                        relaxation2(data, h, tol, N, P, xp, data->pulse.rf_end, data->seq.te, NULL);
                }


                if (get_signal)
                        collect_signal(data, N, P, mxy, sa_r1, sa_r2, sa_b1, xp);


                // Smooth spoiling for FLASH sequences

                if (    (SEQ_FLASH == data->seq.seq_type) ||
                        (SEQ_IRFLASH == data->seq.seq_type))

                        data->tmp.r2spoil = 10000.;


                // Balance z-gradient for bSSFP type sequences

                if (    (SEQ_BSSFP == data->seq.seq_type) ||
                        (SEQ_IRBSSFP == data->seq.seq_type)) {

                        relaxation2(data, h, tol, N, P, xp, data->seq.te, data->seq.tr-data->pulse.rf_end, NULL);

                        data->grad.mom = -data->grad.mom_sl;
                        relaxation2(data, h, tol, N, P, xp, data->seq.tr-data->pulse.rf_end, data->seq.tr, NULL);
                        data->grad.mom = 0.;

                } else {

                        relaxation2(data, h, tol, N, P, xp, data->seq.te, data->seq.tr, NULL);
                }
                data->tmp.r2spoil = 0.;	// effects spoiled sequences only

                break;

        case SIM_STM:
                ;
                // Evolution: 0 -> TE
                apply_sim_matrix(N * P + 1, xstm, mte);

                // Save data
                stm2ode(N, P, xp, xstm);
                collect_signal(data, N, P, mxy, sa_r1, sa_r2, sa_b1, xp);

                // Evolution: TE -> TR
                apply_sim_matrix(N * P + 1, xstm, mtr);

                break;
        }
}


/* ------------ Sequence Specific Blocks -------------- */

void inversion(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end)
{
	struct sim_data inv_data = *data;

        // Enforce ODE: Way more efficient here!
        inv_data.seq.type = SIM_ODE;

        if (data->seq.perfect_inversion) {

                // Apply perfect inversion

                for (int p = 0; p < P; p++)
                        bloch_excitation2(xp[p], xp[p], M_PI, 0.);

                relaxation2(&inv_data, h, tol, N, P, xp, st, end, NULL);

        } else {
                // Hyperbolic Secant inversion

                inv_data.pulse.hs = hs_pulse_defaults;
                inv_data.pulse.hs.on = true;
                inv_data.pulse.hs.duration = data->seq.inversion_pulse_length;
                inv_data.pulse.rf_end = data->seq.inversion_pulse_length;

                start_rf_pulse(&inv_data, h, tol, N, P, xp, NULL);

                // Spoiler gradients
                inv_data.tmp.r2spoil = 10000.;
                relaxation2(&inv_data, h, tol, N, P, xp, st, end, NULL);
        }
}


static void alpha_half_preparation(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N])
{
	struct sim_data prep_data = *data;

        // Enforce ODE: Way more efficient here!
        prep_data.seq.type = SIM_ODE;

        prep_data.pulse.flipangle = data->pulse.flipangle / 2.;
        prep_data.pulse.phase = M_PI;
        prep_data.seq.te = data->seq.prep_pulse_length;
        prep_data.seq.tr = data->seq.prep_pulse_length;

        assert(prep_data.pulse.rf_end <= prep_data.seq.prep_pulse_length);

        prepare_sim(&prep_data, N, P, NULL, NULL);

        run_sim(&prep_data, NULL, NULL, NULL, NULL, h, tol, N, P, xp, NULL, NULL, NULL, false);
}


/* ------------ Main Simulation -------------- */

void bloch_simulation(struct sim_data* data, float (*mxy_sig)[3], float (*sa_r1_sig)[3], float (*sa_r2_sig)[3], float (*sa_m0_sig)[3], float (*sa_b1_sig)[3])
{
	float tol = 10E-6;      // Tolerance of ODE solver

        enum { N = 3 };         // Number of dimensions (x, y, z)
	enum { P = 4 };         // Number of parameters with estimated derivative (Mxy, R1, R2, B1)

        enum { M = N * P + 1 };     // STM based on single vector and additional +1 for linearized system matrix

        long storage_size = data->seq.spin_num * data->seq.rep_num * 3 * sizeof(float);

	float* mxy = xmalloc(storage_size);
	float* sa_r1 = xmalloc(storage_size);
	float* sa_r2 = xmalloc(storage_size);
	float* sa_b1 = xmalloc(storage_size);

	float w_backup = data->voxel.w;
	float zgradient_max = data->grad.mom_sl;

	for (data->tmp.spin_counter = 0; data->tmp.spin_counter < data->seq.spin_num; data->tmp.spin_counter++) {

                float h = 0.0001;

                // Full Symmetric slice profile
		//      - Calculate slice profile by looping over spins with z-gradient
		if (1 != data->seq.spin_num) {

                        // Ensures central spin on main lope is set
			assert(1 == data->seq.spin_num % 2);

			data->grad.mom_sl = zgradient_max / (data->seq.spin_num-1) * (data->tmp.spin_counter - (int)(data->seq.spin_num / 2.));
		}

                // ODE
		float xp[P][N] = { { 0., 0., 1. }, { 0. }, { 0. }, { 0. } };

                // STM
                float xstm[M] = { 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1. };


                // Reset parameters
		data->voxel.w = w_backup;
		data->pulse.phase = 0;
                data->tmp.t = 0;
                data->tmp.rep_counter = 0;

                // Apply perfect inversion

                if (    (SEQ_IRBSSFP == data->seq.seq_type) ||
                        (SEQ_IRFLASH == data->seq.seq_type))
                        inversion(data, h, tol, N, P, xp, 0., data->seq.inversion_spoiler);

                // Alpha/2 and TR/2 signal preparation

                if (    (SEQ_BSSFP == data->seq.seq_type) ||
                        (SEQ_IRBSSFP == data->seq.seq_type))
                        alpha_half_preparation(data, h, tol, N, P, xp);

                // if (STM == data->seq.type) printf("test\n");

                float mte[M][M];
                float mte2[M][M];
                float mtr[M][M];

                ode2stm(N, P, xstm, xp);

                // STM requires two matrices for RFPhase=0 and RFPhase=PI
                // Therefore mte and mte2 need to be estimated
                // FIXME: Do not estimate mtr twice
                if (    (SEQ_BSSFP == data->seq.seq_type) ||
                        (SEQ_IRBSSFP == data->seq.seq_type)) {

                        data->pulse.phase = M_PI;
                        prepare_sim(data, N, P, mte2, mtr);
                        data->pulse.phase = 0.;

                        prepare_sim(data, N, P, mte, mtr);

                } else {

                        prepare_sim(data, N, P, mte, mtr);
                }

                // Loop over Pulse Blocks

                data->tmp.t = 0;

                while (data->tmp.rep_counter < data->seq.rep_num) {

                        // Change phase of bSSFP sequence in each repetition block
                        if (    (SEQ_BSSFP == data->seq.seq_type) ||
                                (SEQ_IRBSSFP == data->seq.seq_type)) {

                                data->pulse.phase = M_PI * (float)(data->tmp.rep_counter);

                                run_sim(data, mxy, sa_r1, sa_r2, sa_b1, h, tol, N, P, xp, xstm, ((0 == data->tmp.rep_counter % 2) ? mte : mte2), mtr, true);

                        } else {

                                run_sim(data, mxy, sa_r1, sa_r2, sa_b1, h, tol, N, P, xp, xstm, mte, mtr, true);
                        }

                        data->tmp.rep_counter++;
                }
	}


	// Sum up magnetization

        sum_up_signal(data, mxy, sa_r1, sa_r2, sa_b1, mxy_sig, sa_r1_sig, sa_r2_sig, sa_m0_sig, sa_b1_sig);

	free(mxy);
	free(sa_r1);
	free(sa_r2);
	free(sa_b1);
}