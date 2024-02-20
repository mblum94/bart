struct config_raki_s {

	_Bool gpu;

	_Bool joined;
	_Bool grappa;

	long ker_dims[3][3];
	long channels[2];

	long epochs;

	struct nn_weights_s* weights;
};

extern struct config_raki_s raki_default;

struct nn_s;

extern const struct nn_s* raki_network_create(const struct config_raki_s* config, long idims[4], long adims[3], _Bool apply);

extern void raki_train(struct config_raki_s* config, long adims[3], long dims[4], long strs[4], const _Complex float* ac_data);
extern void raki_apply(struct config_raki_s* config, long adims[3],
		long dims[4], long strs[4], _Complex float* dst, const _Complex float* src,
		long pdims[4], long pstrs[4], const _Complex float* pat);