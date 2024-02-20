/* Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */
 
#include <stdbool.h>
#include <complex.h>

#include "iter/italgos.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/iovec.h"

#include "iter/iter6.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"

#include "nn/nn.h"
#include "nn/chain.h"
#include "nn/activation.h"
#include "nn/activation_nn.h"
#include "nn/layers_nn.h"
#include "nn/losses.h"
#include "nn/weights.h"




#include "raki.h"



struct config_raki_s raki_default = {

	.weights = NULL,
	.gpu = false,
	.joined = true,
	.ker_dims = { { 3, 3, 3} ,
		   { 1, 1, 1} ,
		   { 2, 2, 2}  },

	.grappa = false,
	.channels = { 128, 128 },
	.epochs = 50,
};

static void compute_receptive_field(const struct config_raki_s* config, long rfield[3])
{

	for (int j = 0; j < 3; j++)
		rfield[j] = 1;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			rfield[j] += config->ker_dims[i][j] - 1;
}

static nn_t raki_network_single_create(const struct config_raki_s* config, long _idims[4], long _adims[3], bool apply)
{
	unsigned int perm[5] = { 3, 0, 1, 2, 4 };
	long idims[5] = { _idims[0], _idims[1], _idims[2], _idims[3], 1 };

	long adims[3];
	md_select_dims(3, md_nontriv_dims(3, _adims), adims, _adims);

	auto lop_net = linop_permute_create(5, perm, idims);
	lop_net = linop_reshape_in_F(lop_net, 4, idims);

	auto conv_init = init_kaiming_create(in_flag_conv(true), false, false, 0);

	if (config->grappa) {

		long rfield[3];
		compute_receptive_field(config, rfield);

		auto net = nn_from_nlop_F(nlop_from_linop_F(lop_net));
		net = nn_append_convcorr_layer(net, 0, NULL, NULL, idims[COIL_DIM] * (config->joined ? md_calc_size(3, adims) - 1 : 1), rfield, false, PAD_VALID, true, apply ? adims : NULL, adims, initializer_clone(conv_init));
		net = nn_append_singleton_dim_in_F(net, 1, NULL);
		return net;
	}

	auto net = nn_from_nlop_F(nlop_from_linop_F(lop_net));
	net = nn_append_convcorr_layer(net, 0, NULL, NULL, config->channels[0], config->ker_dims[0], false, PAD_VALID, true, apply ? adims : NULL, adims, initializer_clone(conv_init));
	net = nn_append_activation(net, 0, NULL, ACT_RELU);
	net = nn_append_convcorr_layer(net, 0, NULL, NULL, config->channels[1], config->ker_dims[1], false, PAD_VALID, true, NULL, adims, initializer_clone(conv_init));
	net = nn_append_activation(net, 0, NULL, ACT_RELU);
	net = nn_append_convcorr_layer(net, 0, NULL, NULL, idims[COIL_DIM] * (config->joined ? md_calc_size(3, adims) - 1 : 1), config->ker_dims[2], false, PAD_VALID, true, NULL, apply ? NULL : adims, initializer_clone(conv_init));

	for (int i = 1; i < 4; i++)
		net = nn_append_singleton_dim_in_F(net, i, NULL);
	
	nn_debug(DP_DEBUG3, net);
	
	return net;
}


const struct nn_s* raki_network_create(const struct config_raki_s* config, long idims[4], long _adims[3], bool apply)
{
	long adims[3];
	md_select_dims(3, md_nontriv_dims(3, _adims), adims, _adims);

	auto result = raki_network_single_create(config, idims, adims, apply);

	if (! config->joined) {

		for (int i = 1; i < md_calc_size(3, adims) - 1; i++) {

			result = nn_combine_FF(result, raki_network_single_create(config, idims, adims, apply));
			
			for (int i = 3; i > 0; i--)
				result = nn_stack_inputs_F(result, i, NULL, 4 + i, NULL, -1);
			
			result = nn_dup_F(result, 0, NULL, 4, NULL);
			result = nn_stack_outputs_F(result, 0, NULL, 1, NULL, 0);
		}
	}

	auto iov = nn_generic_codomain(result, 0, NULL);
	
	unsigned int perm[5] = { 1, 2, 3, 0, 4};
	auto lop_post = linop_permute_create(iov->N, perm, iov->dims);


	long odims[7];
	
	md_copy_dims(5, odims, linop_codomain(lop_post)->dims);
	odims[3] = idims[COIL_DIM];
	odims[4] = md_calc_size(3, adims) - 1;

	lop_post = linop_reshape_out_F(lop_post, 5, odims);
	lop_post = linop_chain_FF(lop_post, linop_padding_create_onedim(5, odims, PAD_SAME, 4, 1, 0));
	
	md_copy_dims(3, odims + 4, adims);
	lop_post = linop_reshape_out_F(lop_post, 7, odims);

	if (apply) {

		unsigned int perm[7] = { 4, 0, 5, 1, 6, 2, 3 };
		lop_post = linop_chain_FF(lop_post, linop_permute_create(7, perm, odims));

		for (int i = 0; i < 3; i++)
			odims[i] *= odims[4 + i];
		
		lop_post = linop_reshape_out_F(lop_post, 4, odims);		
	}

	result = nn_chain2_FF(result, 0, NULL, nn_from_linop_F(lop_post), 0, NULL);
	
	return result;
}


void raki_train(struct config_raki_s* config, long adims[3], long dims[4], long strs[4], const complex float* ac_data)
{
	auto net = raki_network_create(config, dims, adims, false);
	auto iov = nn_generic_codomain(net, 0, NULL);
	
	net = nn_chain2_FF(net, 0, NULL, nn_from_nlop_F(nlop_mse_create(iov->N, iov->dims, ~0)), 0, NULL);

	debug_printf(DP_INFO, "Train RAKI\n");
	nn_debug(DP_INFO, net);

	auto weights = nn_weights_create_from_nn(net);
	nn_init(net, weights);

	if (config->gpu)
		move_gpu_nn_weights(weights);

	auto r_iov = nn_generic_domain(net, 0, NULL);
	auto i_iov = nn_generic_domain(net, 1, NULL);

	complex float* ref = md_alloc_sameplace(r_iov->N, r_iov->dims, CFL_SIZE, weights->tensors[0]);
	complex float* inp = md_alloc_sameplace(i_iov->N, i_iov->dims, CFL_SIZE, weights->tensors[0]);

	md_copy2(i_iov->N, i_iov->dims, MD_STRIDES(i_iov->N, i_iov->dims, CFL_SIZE), inp, strs, ac_data, CFL_SIZE);

	long pos[3];
	compute_receptive_field(config, pos);
	for (int i = 0; i < 3; i++)
		pos[i] = ((pos[i] - 1) / 2) * adims[i]; 

	ac_data = &(MD_ACCESS(3, strs, pos, ac_data));

	md_singleton_strides(3, pos);

	do {

		md_copy2(4, r_iov->dims, r_iov->strs, &(MD_ACCESS(3, r_iov->strs + 4, pos, ref)), strs, &(MD_ACCESS(3, strs, pos, ac_data)), CFL_SIZE);

	} while (md_next(3, r_iov->dims + 4, ~0, pos));

	int NO = nn_get_nr_out_args(net);
	int NI = nn_get_nr_in_args(net);

	assert(1 == NO);

	enum OUT_TYPE out_type[1] = { OUT_OPTIMIZE };
	enum IN_TYPE in_type[NI];

	nn_get_in_types(net, NI, in_type);
	in_type[0] = IN_STATIC;
	in_type[1] = IN_STATIC;

	float* args[NI];
	args[0] = (float*)ref;
	args[1] = (float*)inp;

	for (int i = 0; i < weights->N; i++)
		args[2 + i] = (float*)weights->tensors[i];

	struct iter6_iPALM_conf conf_ipalm = iter6_iPALM_conf_defaults;
	conf_ipalm.INTERFACE.learning_rate = 1.e9;
	conf_ipalm.INTERFACE.epochs = config->epochs;

	const struct operator_p_s* prox[5] = { NULL, NULL, NULL, NULL, NULL };

	iter6_by_conf((struct iter6_conf_s*)(&conf_ipalm), nn_get_nlop(net), NI, in_type, prox, args, NO, out_type, 1, 1, NULL, NULL);

	assert(NULL == config->weights);
	config->weights = weights;
	
	nn_free(net);
	md_free(inp);
	md_free(ref);
}


void raki_apply(struct config_raki_s* config, long adims[3],
		long dims[4], long strs[4], complex float* dst, const complex float* src,
		long pdims[4], long pstrs[4], const complex float* pat)
{
	auto net = raki_network_create(config, dims, adims, true);

	debug_printf(DP_INFO, "Apply RAKI\n");
	nn_debug(DP_INFO, net);

	long pos[3];
	compute_receptive_field(config, pos);
	for (int i = 0; i < 3; i++)
		pos[i] = ((pos[i] - 1) / 2) * adims[i]; 
	
	net = nn_get_wo_weights_F(net, config->weights, true);

	auto oiov = nn_generic_codomain(net, 0, NULL);
	auto iiov = nn_generic_domain(net, 0, NULL);

	nlop_generic_apply2_sameplace(nn_get_nlop(net),
				      1, (int[1]){ oiov->N }, (const long* [1]){ oiov->dims }, (const long* [1]){ strs }, (complex float* [1]){ &(MD_ACCESS(3, strs, pos, dst)) },
				      1, (int[1]){ iiov->N }, (const long* [1]){ iiov->dims }, (const long* [1]){ strs }, &src,
				      config->weights->tensors[0]);
	
	nn_free(net);
	
	complex float* ipat = md_alloc(4, pdims, CFL_SIZE);
	md_zsadd2(4, pdims, MD_STRIDES(4, pdims, CFL_SIZE), ipat, pstrs, pat, -1);
	md_zsmul(4, pdims, ipat, ipat, -1);

	md_zmul2(4, dims, strs, dst, strs, dst, MD_STRIDES(4, pdims, CFL_SIZE), ipat);
	md_free(ipat);

	md_zfmac2(4, dims, strs, dst, strs, src, pstrs, pat);
}