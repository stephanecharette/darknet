// Darknet Next Gen - Darknet YOLO framework for computer vision / object detection.
// MIT license applies.  See "license.txt" for details.

#pragma once

#include "darknet-ng.hpp"


namespace Darknet_ng
{
#ifdef OLD_UNUSED /// @todo remove?
	struct Section // was: section
	{
		char *type;
		list *options;
		MStr options;
	};
#endif

	struct ContrastiveParams // was: contrastive_params
	{
		float	sim;
		float	exp_sim;
		float	P;
		int		i;
		int		j;
		int		time_step_i;
		int		time_step_j;
	};

	struct Tree // was: tree
	{
		int *leaf;
		int n;
		int *parent;
		int *child;
		int *group;
		char **name;

		int groups;
		int *group_size;
		int *group_offset;
	};

	struct UpdateArgs
	{
		int		batch;
		float	learning_rate;
		float	momentum;
		float	decay;
		int		adam;
		float	B1;
		float	B2;
		float	eps;
		int		t;
	};

	struct NetworkState /// was: network_state
	{
		float *truth;
		float *input;
		float *delta;
		float *workspace;
		int train;
		int index;
#if 0 /// @todo
		Network net;
#endif
	};

	struct Box /// was: box
	{
		float x;
		float y;
		float w;
		float h;
	};

	struct Detection /// was: detection
	{
		Box bbox;
		int classes;
		int best_class_idx;
		float *prob;
		float *mask;
		float objectness;
		int sort_class;
		float *uc; // Gaussian_YOLOv3 - tx,ty,tw,th uncertainty
		int points; // bit-0 - center, bit-1 - top-left-corner, bit-2 - bottom-right-corner
		float *embeddings;  // embeddings for tracking
		int embedding_size;
		float sim;
		int track_id;
	};
}
