/* Copyright 2023. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *
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
#include "num/fft.h"
#include "num/init.h"

#include "noncart/nufft.h"
#include "linops/linop.h"

#include "noir/recon2.h"
#include "noir/misc.h"


#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif



static const char help_str[] =
		"Estimate coil sensitivities using ENLIVE calibration.";





int main_ncalib(int argc, char* argv[argc])
{
	double start_time = timestamp();

	const char* ksp_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_OUTFILE(true, &out_file, "sensitivities"),
	};

	struct noir2_conf_s conf = noir2_defaults;
	conf.iter = 12;

	long calsize[3] = { 32, 32, 32 };
	long ksenssize[3] = { 16, 16, 16 };

	long my_sens_dims[3] = { 0, 0, 0 };

	const char* pat_file = NULL;
	const char* trj_file = NULL;
	const char* bas_file = NULL;

	int maps = 1;
	bool normalize = false;
	float scaling = 0;
	float oversampling_coils = 0;

	const struct opt_s opts[] = {

		OPT_SET('g', &bart_use_gpu, "use gpu"),
		OPT_INFILE('t', &trj_file, "file", "kspace trajectory"),
		OPT_INFILE('p', &pat_file, "file", "kspace pattern"),
		OPT_INFILE('B', &bas_file, "file", "subspace basis"),

		OPT_UINT('i', &conf.iter, "iter", "Number of Newton steps"),
		OPT_FLOAT('R', &conf.redu, "", "(reduction factor)"),
		OPTL_INT(0, "cgiter", &conf.cgiter, "iter", "(iterations for linearized problem)"),
		OPTL_FLOAT(0, "cgtol", &conf.cgtol, "tol", "(tolerance for linearized problem)"),
	
		OPTL_FLOAT(0, "alpha", &conf.alpha, "val", "(alpha in first iteration)"),
		OPT_FLOAT('M', &conf.alpha_min, "", "(minimum for regularization)"),
		OPT_FLOAT('a', &conf.a, "", "(a in 1 + a * \\Laplace^-b/2)"),
		OPT_FLOAT('b', &conf.b, "", "(b in 1 + a * \\Laplace^-b/2)"),
		OPT_FLOAT('w', &scaling, "", "(inverse scaling of the data)"),

		OPT_SET('N', &normalize, "Normalize coil sensitivities"),
		OPT_INT('m', &maps, "nmaps", "Number of ENLIVE maps to use in reconstruction"),	
		OPTL_VEC3('x', "dims", &my_sens_dims, "x:y:z", "Explicitly specify sens dimensions"),
		OPTL_FLOAT(0, "sens-os", &(oversampling_coils), "val", "(over-sampling factor for sensitivities)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);


	conf.gpu = bart_use_gpu;
	num_init_gpu_support();

	long ksp_dims[DIMS];
	complex float* kspace = load_cfl(ksp_file, DIMS, ksp_dims);

	// The only multimap we understand with is the one we do ourselves, where
	// we allow multiple images and sensitivities during the reconstruction
	assert(1 == ksp_dims[MAPS_DIM]);

	if (1 != ksp_dims[SLICE_DIM]) {

		debug_printf(DP_INFO, "SMS-NLINV reconstruction. Multiband factor: %d\n", ksp_dims[SLICE_DIM]);
		conf.sms = true;
	}

	complex float* pattern = NULL;
	long pat_dims[DIMS];

	if (NULL != pat_file) {

		pattern = load_cfl(pat_file, DIMS, pat_dims);
	} else {

		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
		pattern = anon_cfl("", DIMS, pat_dims);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace);
	}

	const complex float* basis = NULL;
	long bas_dims[DIMS];

	if (NULL != bas_file) {

		basis = load_cfl(bas_file, DIMS, bas_dims);
	} else {

		md_singleton_dims(DIMS, bas_dims);
	}

	long dims[DIMS];
	md_copy_dims(DIMS, dims, ksp_dims);

	long sens_dims[DIMS];

	long trj_dims[DIMS];
	complex float* traj  = NULL;

	if (NULL != trj_file) {

		conf.noncart = true;

		traj = load_cfl(trj_file, DIMS, trj_dims);

		long tdims[DIMS];
		estimate_im_dims(DIMS, FFT_FLAGS, tdims, trj_dims, traj);

		md_select_dims(3, md_nontriv_dims(3, tdims), dims, calsize);
		md_copy_dims(DIMS - 3, dims + 3, ksp_dims + 3);

		if (0 == md_calc_size(3, my_sens_dims)) {

			md_copy_dims(3, my_sens_dims, tdims);
			debug_printf(DP_INFO, "Est. image size: %ld %ld %ld\n", my_sens_dims[0], my_sens_dims[1], my_sens_dims[2]);
		}

		// discard high frequencies (needed for periodic in toeplitz)

		complex float* trj_tmp = md_alloc(DIMS, trj_dims, CFL_SIZE);

		md_zabs(DIMS, trj_dims, trj_tmp, traj);

		long cord_dims[DIMS];
		md_select_dims(DIMS, MD_BIT(0), cord_dims, trj_dims);
		
		complex float inv_dims[3] = { 1. / dims[0],  1. / dims[1], 1. / dims[2]};
		md_zmul2(DIMS, trj_dims, MD_STRIDES(DIMS, trj_dims, CFL_SIZE), trj_tmp, MD_STRIDES(DIMS, trj_dims, CFL_SIZE), trj_tmp, MD_STRIDES(DIMS, cord_dims, CFL_SIZE), inv_dims);

		md_zslessequal(DIMS, trj_dims, trj_tmp, trj_tmp, 0.5);

		for (int i = 0; i < trj_dims[0]; i++)
			md_zmul2(DIMS, pat_dims, MD_STRIDES(DIMS, pat_dims, CFL_SIZE), pattern, MD_STRIDES(DIMS, pat_dims, CFL_SIZE), pattern, MD_STRIDES(DIMS, trj_dims, CFL_SIZE), trj_tmp + i);

		md_free(trj_tmp);

	} else {

		assert(0 == md_calc_size(3, my_sens_dims));
		md_copy_dims(3, my_sens_dims, ksp_dims);

		long nksp_dims[DIMS];
		long npat_dims[DIMS];
		md_copy_dims(DIMS, nksp_dims, ksp_dims);
		md_copy_dims(DIMS, npat_dims, pat_dims);

		for (int i = 0; i < 3; i++) {

			nksp_dims[i] = MIN(nksp_dims[i], calsize[i]);
			npat_dims[i] = MIN(npat_dims[i], calsize[i]);
		}

		complex float* nksp = anon_cfl(NULL, DIMS, nksp_dims);
		complex float* npat = anon_cfl(NULL, DIMS, npat_dims);

		md_resize_center(DIMS, nksp_dims, nksp, ksp_dims, kspace, CFL_SIZE);
		md_resize_center(DIMS, npat_dims, npat, pat_dims, pattern, CFL_SIZE);

		unmap_cfl(DIMS, ksp_dims, kspace);
		unmap_cfl(DIMS, pat_dims, pattern);

		kspace = nksp;
		pattern = npat;

		md_copy_dims(DIMS, ksp_dims, nksp_dims);
		md_copy_dims(DIMS, pat_dims, npat_dims);

		md_copy_dims(DIMS, dims, ksp_dims);
	}

	// for ENLIVE maps
	dims[MAPS_DIM] = maps;

	if (NULL != basis) {

		assert(1 == ksp_dims[COEFF_DIM]);
		assert(bas_dims[TE_DIM] == ksp_dims[TE_DIM]);
		
		if (conf.noncart)
			assert(1 == md_calc_size(5, bas_dims));
		else
			md_check_compat(5, ~0, bas_dims, dims);

		dims[COEFF_DIM] = bas_dims[COEFF_DIM];
		dims[TE_DIM] = 1;
	}

	md_copy_dims(DIMS, sens_dims, dims);
	md_copy_dims(3, sens_dims, my_sens_dims);

	if (0 == scaling)
		conf.scaling = -100;
	else
		conf.scaling = scaling;

	if (0 == oversampling_coils)
		conf.oversampling_coils = (1 == maps) ? 1.25 : 1;
	else
	 	conf.oversampling_coils = oversampling_coils;

	long ksens_dims[DIMS];
	md_copy_dims(DIMS, ksens_dims, sens_dims);
	md_select_dims(3, md_nontriv_dims(3, sens_dims), ksens_dims, ksenssize);

	long img_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, dims);

	long cim_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, dims);

	complex float* img = md_alloc(DIMS, img_dims, CFL_SIZE);
	complex float* ksens = md_alloc(DIMS, ksens_dims, CFL_SIZE);
	complex float* sens = create_cfl(out_file, DIMS, sens_dims);

	md_zfill(DIMS, img_dims, img, 1.);
	md_clear(DIMS, ksens_dims, ksens, CFL_SIZE);


	if (NULL != traj) {

		struct nufft_conf_s nufft_conf = nufft_conf_defaults;
		nufft_conf.toeplitz = true;
		nufft_conf.pcycle = false;
		nufft_conf.periodic = false;
		nufft_conf.cache_psf_grdding = true;
		conf.nufft_conf = &nufft_conf;

		noir2_recon_noncart(&conf, DIMS,
			img_dims, img, NULL,
			sens_dims, sens,
			ksens_dims, ksens, NULL,
			ksp_dims, kspace,
			trj_dims, traj,
			pat_dims, pattern,
			bas_dims, basis,
			MD_SINGLETON_DIMS(DIMS), NULL,
			cim_dims);

	} else {

		noir2_recon_cart(&conf, DIMS,
			img_dims, img, NULL,
			sens_dims, sens,
			ksens_dims, ksens, NULL,
			ksp_dims, kspace,
			pat_dims, pattern,
			bas_dims, basis,
			MD_SINGLETON_DIMS(DIMS), NULL,
			cim_dims);
	}

	md_free(img);
	md_free(ksens);

	unmap_cfl(DIMS, ksp_dims, kspace);

	if (NULL != traj)
		unmap_cfl(DIMS, trj_dims, traj);

	unmap_cfl(DIMS, pat_dims, pattern);

	if (normalize) {

		long scl_dims[DIMS];
		md_select_dims(DIMS, ~(MAPS_FLAG | COIL_FLAG), scl_dims, sens_dims);

		complex float* scl = md_alloc(DIMS, scl_dims, CFL_SIZE);

		md_zrss(DIMS, sens_dims, (MAPS_FLAG | COIL_FLAG), scl, sens);
		md_zdiv2(DIMS, sens_dims, MD_STRIDES(DIMS, sens_dims, CFL_SIZE), sens, MD_STRIDES(DIMS, sens_dims, CFL_SIZE), sens, MD_STRIDES(DIMS, scl_dims, CFL_SIZE), scl);

		md_free(scl);
	}

	unmap_cfl(DIMS, sens_dims, sens);

	if (NULL != basis)
		unmap_cfl(DIMS, bas_dims, basis);

	double recosecs = timestamp() - start_time;

	debug_printf(DP_DEBUG2, "Total time: %.2f s\n", recosecs);

	return 0;
}


