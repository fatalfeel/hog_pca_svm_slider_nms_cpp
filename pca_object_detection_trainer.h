#ifndef DLIB_PCA_OBJECT_DETECTION_TRAiNER_Hh_
#define DLIB_PCA_OBJECT_DETECTION_TRAiNER_Hh_

namespace dlib
{
	// ----------------------------------------------------------------------------------------
	template <typename image_scanner_type>
	class pca_object_detection_trainer : noncopyable
	{
	public:
		typedef double scalar_type;
		typedef default_memory_manager mem_manager_type;
		typedef object_detector<image_scanner_type> trained_function_type;

		explicit pca_object_detection_trainer(const image_scanner_type& scanner_)
		{
			C = 1;
			verbose = false;
			eps = 0.1;
			num_threads = 2;
			max_cache_size = 5;
			match_eps = 0.5;
			loss_per_missed_target = 1;
			loss_per_false_alarm = 1;

			scanner.copy_configuration(scanner_);

			auto_overlap_tester = true;
		}

		const image_scanner_type& get_scanner() const
		{
			return scanner;
		}

		bool auto_set_overlap_tester() const
		{
			return auto_overlap_tester;
		}

		void set_overlap_tester(const test_box_overlap& tester)
		{
			overlap_tester = tester;
			auto_overlap_tester = false;
		}

		test_box_overlap get_overlap_tester() const
		{
			return overlap_tester;
		}

		void set_num_threads (unsigned long num)
		{
			num_threads = num;
		}

		unsigned long get_num_threads() const
		{
			return num_threads;
		}

		void set_epsilon(scalar_type eps_)
		{
			eps = eps_;
		}

		scalar_type get_epsilon() const
		{
			return eps;
		}

		void set_max_runtime(const std::chrono::nanoseconds& max_runtime)
		{
			solver.set_max_runtime(max_runtime);
		}

		std::chrono::nanoseconds get_max_runtime() const
		{
			return solver.get_max_runtime();
		}

		void set_max_cache_size (unsigned long max_size)
		{
			max_cache_size = max_size;
		}

		unsigned long get_max_cache_size () const
		{
			return max_cache_size;
		}

		void be_verbose ()
		{
			verbose = true;
		}

		void be_quiet ()
		{
			verbose = false;
		}

		void set_oca (const oca& item)
		{
			solver = item;
		}

		const oca get_oca () const
		{
			return solver;
		}

		void set_c (scalar_type C_)
		{
			C = C_;
		}

		scalar_type get_c () const
		{
			return C;
		}

		void set_match_eps (double eps)
		{
			match_eps = eps;
		}

		double get_match_eps () const
		{
			return match_eps;
		}

		double get_loss_per_missed_target () const
		{
			return loss_per_missed_target;
		}

		void set_loss_per_missed_target (double loss)
		{
			loss_per_missed_target = loss;
		}

		double get_loss_per_false_alarm () const
		{
			return loss_per_false_alarm;
		}

		void set_loss_per_false_alarm (double loss)
		{
			loss_per_false_alarm = loss;
		}

		template <typename image_array_type>
		const trained_function_type train (	const image_array_type& images,
											const std::vector<std::vector<full_object_detection> >& truth_object_detections) const
		{
			std::vector<std::vector<rectangle> > empty_ignore(images.size());
			return train_impl(images, truth_object_detections, empty_ignore, test_box_overlap());
		}

		template <typename image_array_type>
		const trained_function_type train ( const image_array_type& images,
											const std::vector<std::vector<full_object_detection> >& truth_object_detections,
											const std::vector<std::vector<rectangle> >& ignore,
											const test_box_overlap& ignore_overlap_tester = test_box_overlap() ) const
		{
			return train_impl(images, truth_object_detections, ignore, ignore_overlap_tester);
		}

		template <typename image_array_type>
		const trained_function_type train ( const image_array_type& images,
											const std::vector<std::vector<rectangle> >& truth_object_detections	) const
		{
			std::vector<std::vector<rectangle> > empty_ignore(images.size());
			return train(images, truth_object_detections, empty_ignore, test_box_overlap());
		}

		template <typename image_array_type>
		const trained_function_type train ( const image_array_type& images,
											const std::vector<std::vector<rectangle> >& truth_object_detections,
											const std::vector<std::vector<rectangle> >& ignore,
											const test_box_overlap& ignore_overlap_tester = test_box_overlap() ) const
		{
			std::vector<std::vector<full_object_detection> > truth_dets(truth_object_detections.size());
			for (unsigned long i = 0; i < truth_object_detections.size(); ++i)
			{
				for (unsigned long j = 0; j < truth_object_detections[i].size(); ++j)
				{
					truth_dets[i].push_back(full_object_detection(truth_object_detections[i][j]));
				}
			}

			return train_impl(images, truth_dets, ignore, ignore_overlap_tester);
		}

	private:
		template <typename image_array_type>
		const trained_function_type train_impl (const image_array_type& images,
												const std::vector<std::vector<full_object_detection> >& truth_object_detections,
												const std::vector<std::vector<rectangle> >& ignore,
												const test_box_overlap& ignore_overlap_tester) const
		{
			pca_svm_object_detection_problem<image_scanner_type,image_array_type>svm_prob(	scanner,
																																									overlap_tester,
																																									auto_overlap_tester,
																																									images,
																																									truth_object_detections,
																																									ignore,
																																									ignore_overlap_tester,
																																									num_threads  );

			if (verbose)
				svm_prob.be_verbose();

			svm_prob.set_c(C);
			svm_prob.set_epsilon(eps);
			svm_prob.set_max_cache_size(max_cache_size);
			svm_prob.set_match_eps(match_eps);
			svm_prob.set_loss_per_missed_target(loss_per_missed_target);
			svm_prob.set_loss_per_false_alarm(loss_per_false_alarm);
			configure_nuclear_norm_regularizer(scanner, svm_prob);
			matrix<double,0,1> w;

			// Run the optimizer to find the optimal w.
			solver(svm_prob,w);

			// report the results of the training.
			return object_detector<image_scanner_type>(scanner, svm_prob.get_overlap_tester(), w);
		}

		image_scanner_type scanner;
		test_box_overlap overlap_tester;

		double C;
		oca solver;
		double eps;
		double match_eps;
		bool verbose;
		unsigned long num_threads;
		unsigned long max_cache_size;
		double loss_per_missed_target;
		double loss_per_false_alarm;
		bool auto_overlap_tester;
	};
// ----------------------------------------------------------------------------------------
}

#endif
