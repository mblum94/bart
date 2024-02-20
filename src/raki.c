/* Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "networks/raki.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

struct regular_us_s {

	long acceleration;

	long pos_ac;
	long sze_ac;

	long pos_us[2];
	long sze_us[2];
};

static struct regular_us_s analyze_us(int sze, complex float pat[sze]) {

	struct regular_us_s ret;

	ret.pos_us[0] = 0;
	
	while (0. == pat[ret.pos_us[0]]) {

		ret.pos_us[0]++;
		assert(ret.pos_us[0] < sze);
	}

	ret.acceleration = 1;
	
	while ((1 < sze) && (0. == pat[ret.pos_us[0] + ret.acceleration])) {

		ret.acceleration++;
		assert(ret.pos_us[0] + ret.acceleration < sze);
	}


	long pos_end = ret.pos_us[0];

	while ((pos_end + ret.acceleration < sze) && (0. != pat[pos_end + ret.acceleration]))
		pos_end += ret.acceleration;
	
	ret.sze_us[0] = pos_end - ret.pos_us[0] + 1;



	pos_end = sze - 1;
	
	while (0. == pat[pos_end])
		pos_end --;
	
	ret.pos_us[1] = pos_end;

	while ((ret.pos_us[1] - ret.acceleration >= 0) && (0. != pat[ret.pos_us[1] - ret.acceleration]))
		ret.pos_us[1] -= ret.acceleration;
	
	ret.sze_us[1] = pos_end - ret.pos_us[1] + 1;


	
	ret.pos_ac = ret.pos_us[0];
	while ((ret.pos_ac + 1 < sze) && ((0. == pat[ret.pos_ac]) || (0. == pat[ret.pos_ac + 1])))
		ret.pos_ac += 1;
	
	if ((1 < sze) && (ret.pos_ac == ret.pos_us[0] + ret.sze_us[0] - 1)) {

		error("No AC-Region found!\n");
	} else {

		ret.sze_ac = 1;

		while ((ret.pos_ac + ret.sze_ac < sze) && (0. != pat[ret.pos_ac + ret.sze_ac]))
			ret.sze_ac += 1;
	}
	

	return ret;
}



static const char help_str[] =
		"Completes k-space by the RAKI method.";





int main_raki(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "kspace in"),
		ARG_OUTFILE(true, &out_file, "kspace out"),
	};

	long max_cal_size = 32;
	struct config_raki_s config = raki_default;

	const struct opt_s opts[] = {

		OPTL_SET('g', "gpu", &(bart_use_gpu), "GPU"),
		OPTL_SET(0, "grappa", &(config.grappa), "Only use one convolutional layer (GRAPPA)"),
		OPTL_LONG(0, "max_cal_size", &max_cal_size, "", "restrict AC region (default: 32)"),
		OPTL_LONG(0, "epochs", &(config.epochs), "", "number of epochs for training (default: 50)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init_gpu_support();
	config.gpu = bart_use_gpu;

	int N = DIMS;
	long ksp_dims[N];

	complex float* in_data = load_cfl(in_file, N, ksp_dims);
	complex float* out_data = create_cfl(out_file, N, ksp_dims);

	md_zabs(N, ksp_dims, out_data, in_data);

	complex float scale = 0;
	md_zmax2(N, ksp_dims, MD_SINGLETON_STRS(N), &scale, MD_SINGLETON_STRS(N), &scale, MD_STRIDES(N, ksp_dims, CFL_SIZE), out_data);
	scale /= 0.015;
	md_zsmul(DIMS, ksp_dims, in_data, in_data, 1. / scale);
	

	long pat_dims[N];
	md_select_dims(N, FFT_FLAGS, pat_dims, ksp_dims);
	complex float* pattern = anon_cfl(NULL, N, pat_dims);

	estimate_pattern(N, ksp_dims, COIL_FLAG, pattern, in_data);

	
	struct regular_us_s reg_us[3];

	for (int i = 0; i < 3; i++) {

		complex float* pat_ax = md_alloc(1, MD_DIMS(pat_dims[i]), CFL_SIZE);

		if (1 == pat_dims[i])
			md_zfill(1, MD_DIMS(pat_dims[i]), pat_ax, 1);
		else
			estimate_pattern(3, pat_dims, ~MD_BIT(i), pat_ax, pattern);

		reg_us[i] = analyze_us(pat_dims[i], pat_ax);

		if (reg_us[i].sze_ac > max_cal_size) {

			reg_us[i].pos_ac += (reg_us[i].sze_ac - max_cal_size) / 2;
			reg_us[i].sze_ac = max_cal_size;
		}


		md_free(pat_ax);
	}

	debug_printf(DP_INFO, "Acceleration: %ld x %ld x %ld\n", reg_us[0].acceleration, reg_us[1].acceleration, reg_us[2].acceleration);
	debug_printf(DP_INFO, "Auto-Calibration: %ld x %ld x %ld\n", reg_us[0].sze_ac, reg_us[1].sze_ac, reg_us[2].sze_ac);
	debug_printf(DP_INFO, "at: %ld x %ld x %ld\n", reg_us[0].pos_ac, reg_us[1].pos_ac, reg_us[2].pos_ac);
	
	long acceleration[3] = { reg_us[0].acceleration, reg_us[1].acceleration, reg_us[2].acceleration };

	long ac_dims[4] = { reg_us[0].sze_ac, reg_us[1].sze_ac, reg_us[2].sze_ac, ksp_dims[3] };
	long ac_pos[4] = { reg_us[0].pos_ac, reg_us[1].pos_ac, reg_us[2].pos_ac, 0 };

	for (int i = 0; i < 3; i++)
		md_select_dims(3, md_nontriv_dims(3, ac_dims), config.ker_dims[i], config.ker_dims[i]);

	raki_train(&config, acceleration, ac_dims, MD_STRIDES(4, ksp_dims, CFL_SIZE), &(MD_ACCESS(4, MD_STRIDES(4, ksp_dims, CFL_SIZE), ac_pos, in_data)));

	for (int x = 0; x < ((reg_us[0].pos_us[0] == reg_us[0].pos_us[1]) ? 1 : 2); x++)
		for (int y = 0; y < ((reg_us[1].pos_us[0] == reg_us[1].pos_us[1]) ? 1 : 2); y++)
			for (int z = 0; z < ((reg_us[2].pos_us[0] == reg_us[2].pos_us[1]) ? 1 : 2); z++) {

				long tdims[4] = { reg_us[0].sze_us[x], reg_us[1].sze_us[y], reg_us[2].sze_us[z], ksp_dims[3] };
				long pos[4] = { reg_us[0].pos_us[x], reg_us[1].pos_us[y], reg_us[2].pos_us[z], 0 };
				
				long strs[4];

				long pdims[4];
				long pstrs[4];

				md_select_dims(4, ~COIL_FLAG, pdims, tdims);


				md_calc_strides(4, strs, ksp_dims, CFL_SIZE);
				md_calc_strides(4, pstrs, pat_dims, CFL_SIZE);

				debug_print_dims(DP_INFO, 3, pos);

				raki_apply(&config, acceleration, tdims, strs,
					   &(MD_ACCESS(4, strs, pos, out_data)), &(MD_ACCESS(4, strs, pos, in_data)),
					   pdims, pstrs, &(MD_ACCESS(4, pstrs, pos, pattern)));

			}

	md_zsmul(DIMS, ksp_dims, out_data, out_data, scale);
	
	unmap_cfl(N, ksp_dims, in_data);
	unmap_cfl(N, ksp_dims, out_data);

	if (!config.grappa)
		debug_printf(DP_WARN, "RAKI not final and introduces artifacts in the center of the image!\n");
	
	/**
	bart phantom -s8 -k ksp
	bart upat -Y128 -Z1 -y2 -z1 -c16 pat
	bart fmac ksp pat ksp_us
	bart raki -g --grappa --epochs 100 ksp_us ksp_raki
	bart fft -i -u 7 ksp_raki cim_raki
	bart rss 8 cim_raki rss_raki
	*/

	return 0;
}


