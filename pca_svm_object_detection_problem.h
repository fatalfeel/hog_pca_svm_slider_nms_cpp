// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PCA_SVM_ObJECT_DETECTION_PROBLEM_Hh_
#define DLIB_PCA_SVM_ObJECT_DETECTION_PROBLEM_Hh_

namespace dlib
{
// ----------------------------------------------------------------------------------------
    template <typename image_scanner_type, typename image_array_type>
    class pca_svm_object_detection_problem : public structural_svm_problem_threaded<matrix<double,0,1> >, noncopyable
    {
    public:
        pca_svm_object_detection_problem(const image_scanner_type& scanner,
																		 const test_box_overlap& overlap_tester,
																		 const bool auto_overlap_tester,
																		 const image_array_type& images_,
																		 const std::vector<std::vector<full_object_detection> >& truth_object_detections_,
																		 const std::vector<std::vector<rectangle> >& ignore_,
																		 const test_box_overlap& ignore_overlap_tester_,
																		 unsigned long num_threads = 2):	structural_svm_problem_threaded<matrix<double,0,1> >(num_threads),
																																	boxes_overlap(overlap_tester),
																																	images(images_),
																																	truth_object_detections(truth_object_detections_),
																																	ignore(ignore_),
																																	ignore_overlap_tester(ignore_overlap_tester_),
																																	match_eps(0.5),
																																	loss_per_false_alarm(1),
																																	loss_per_missed_target(1)
        {
            // The purpose of the max_num_dets member variable is to give us a reasonable
            // upper limit on the number of detections we can expect from a single image.
            // This is used in the separation_oracle to put a hard limit on the number of
            // detections we will consider.  We do this purely for computational reasons
            // since otherwise we can end up wasting large amounts of time on certain
            // pathological cases during optimization which ultimately do not influence the
            // result.  Therefore, we force the separation oracle to only consider the
            // max_num_dets strongest detections.
            max_num_dets = 0;
            for (unsigned long i = 0; i < truth_object_detections.size(); ++i)
            {
                if (truth_object_detections[i].size() > max_num_dets)
                    max_num_dets = truth_object_detections[i].size();
            }
            max_num_dets = max_num_dets*3 + 10;

            initialize_scanners(scanner, num_threads);

            if (auto_overlap_tester)
            {
                auto_configure_overlap_tester();
            }
        }

        test_box_overlap get_overlap_tester () const
        {
            return boxes_overlap;
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

    private:
        void auto_configure_overlap_tester()
        {
            std::vector<std::vector<rectangle> > mapped_rects(truth_object_detections.size());
            for (unsigned long i = 0; i < truth_object_detections.size(); ++i)
            {
                mapped_rects[i].resize(truth_object_detections[i].size());
                for (unsigned long j = 0; j < truth_object_detections[i].size(); ++j)
                {
                    mapped_rects[i][j] = scanners[i].get_best_matching_rect(truth_object_detections[i][j].get_rect());
                }
            }

            boxes_overlap = find_tight_overlap_tester(mapped_rects);
        }


        virtual long get_num_dimensions () const
        {
            return scanners[0].get_num_dimensions() + 1;// for threshold
        }

        virtual long get_num_samples () const
        {
            return images.size();
        }

        virtual void get_truth_joint_feature_vector (long idx, feature_vector_type& psi) const
        {
            const image_scanner_type& scanner = scanners[idx];

            psi.set_size(get_num_dimensions());
            std::vector<rectangle> mapped_rects;

            psi = 0;
            for (unsigned long i = 0; i < truth_object_detections[idx].size(); ++i)
            {
                mapped_rects.push_back(scanner.get_best_matching_rect(truth_object_detections[idx][i].get_rect()));
                scanner.get_feature_vector(truth_object_detections[idx][i], psi);
            }
            psi(scanner.get_num_dimensions()) = -1.0*truth_object_detections[idx].size();

            // check if any of the boxes overlap.  If they do then it is impossible for
            // us to learn to correctly classify this sample
            for (unsigned long i = 0; i < mapped_rects.size(); ++i)
            {
                for (unsigned long j = i+1; j < mapped_rects.size(); ++j)
                {
                    if (boxes_overlap(mapped_rects[i], mapped_rects[j]))
                    {
                        const double area_overlap = mapped_rects[i].intersect(mapped_rects[j]).area();
                        const double match_amount = area_overlap/(double)( mapped_rects[i]+mapped_rects[j]).area();
                        const double overlap_amount = area_overlap/std::min(mapped_rects[i].area(),mapped_rects[j].area());

                        using namespace std;
                        ostringstream sout;
                        sout << "An impossible set of object labels was detected. This is happening because ";
                        sout << "the truth labels for an image contain rectangles which overlap according to the ";
                        sout << "test_box_overlap object supplied for non-max suppression.  To resolve this, you ";
                        sout << "either need to relax the test_box_overlap object so it doesn't mark these rectangles as ";
                        sout << "overlapping or adjust the truth rectangles in your training dataset. ";

                        // make sure the above string fits nicely into a command prompt window.
                        string temp = sout.str();
                        sout.str(""); sout << wrap_string(temp,0,0) << endl << endl;


                        sout << "image index: "<< idx << endl;
                        sout << "The offending rectangles are:\n";
                        sout << "rect1: "<< mapped_rects[i] << endl;
                        sout << "rect2: "<< mapped_rects[j] << endl;
                        sout << "match amount:   " << match_amount << endl;
                        sout << "overlap amount: " << overlap_amount << endl;
                        throw dlib::impossible_labeling_error(sout.str());
                    }
                }
            }

            // make sure the mapped rectangles are within match_eps of the
            // truth rectangles.
            for (unsigned long i = 0; i < mapped_rects.size(); ++i)
            {
                const double area = (truth_object_detections[idx][i].get_rect().intersect(mapped_rects[i])).area();
                const double total_area = (truth_object_detections[idx][i].get_rect() + mapped_rects[i]).area();
                if (area/total_area <= match_eps)
                {
                    using namespace std;
                    ostringstream sout;
                    sout << "An impossible set of object labels was detected.  This is happening because ";
                    sout << "none of the object locations checked by the supplied image scanner is a close ";
                    sout << "enough match to one of the truth boxes in your training dataset.  To resolve this ";
                    sout << "you need to either lower the match_eps, adjust the settings of the image scanner ";
                    sout << "so that it is capable of hitting this truth box, or adjust the offending truth rectangle so it ";
                    sout << "can be matched by the current image scanner.  Also, if you ";
                    sout << "are using the scan_fhog_pyramid object then you could try using a finer image pyramid.  ";
                    sout << "Additionally, the scan_fhog_pyramid scans a fixed aspect ratio box across the image when it ";
                    sout << "searches for objects.  So if you are getting this error and you are using the scan_fhog_pyramid, ";
                    sout << "it's very likely the problem is that your training dataset contains truth rectangles of widely ";
                    sout << "varying aspect ratios.  The solution is to make sure your training boxes all have about the same aspect ratio. ";


                    // make sure the above string fits nicely into a command prompt window.
                    string temp = sout.str();
                    sout.str(""); sout << wrap_string(temp,0,0) << endl << endl;

                    sout << "image index              "<< idx << endl;
                    sout << "match_eps:               "<< match_eps << endl;
                    sout << "best possible match:     "<< area/total_area << endl;
                    sout << "truth rect:              "<< truth_object_detections[idx][i].get_rect() << endl;
                    sout << "truth rect width/height: "<< truth_object_detections[idx][i].get_rect().width()/(double)truth_object_detections[idx][i].get_rect().height() << endl;
                    sout << "truth rect area:         "<< truth_object_detections[idx][i].get_rect().area() << endl;
                    sout << "nearest detection template rect:              "<< mapped_rects[i] << endl;
                    sout << "nearest detection template rect width/height: "<< mapped_rects[i].width()/(double)mapped_rects[i].height() << endl;
                    sout << "nearest detection template rect area:         "<< mapped_rects[i].area() << endl;
                    throw dlib::impossible_labeling_error(sout.str());
                }

            }
        }

        virtual void separation_oracle (const long idx,
        								const matrix_type& current_solution,
										scalar_type& loss,
										feature_vector_type& psi) const
        {
            const image_scanner_type& scanner = scanners[idx];

            std::vector<std::pair<double, rectangle> > dets;
            const double thresh = current_solution(scanner.get_num_dimensions());


            scanner.detect(current_solution, dets, thresh-loss_per_false_alarm);


            // The loss will measure the number of incorrect detections.  A detection is
            // incorrect if it doesn't hit a truth rectangle or if it is a duplicate detection
            // on a truth rectangle.
            loss = truth_object_detections[idx].size()*loss_per_missed_target;

            // Measure the loss augmented score for the detections which hit a truth rect.
            std::vector<double> truth_score_hits(truth_object_detections[idx].size(), 0);

            // keep track of which truth boxes we have hit so far.
            std::vector<bool> hit_truth_table(truth_object_detections[idx].size(), false);

            std::vector<rectangle> final_dets;
            // The point of this loop is to fill out the truth_score_hits array.
            for (unsigned long i = 0; i < dets.size() && final_dets.size() < max_num_dets; ++i)
            {
                if (overlaps_any_box(boxes_overlap, final_dets, dets[i].second))
                    continue;

                const std::pair<double,unsigned int> truth = find_best_match(truth_object_detections[idx], dets[i].second);

                final_dets.push_back(dets[i].second);

                const double truth_match = truth.first;
                // if hit truth rect
                if (truth_match > match_eps)
                {
                    // if this is the first time we have seen a detect which hit truth_object_detections[idx][truth.second]
                    const double score = dets[i].first - thresh;
                    if (hit_truth_table[truth.second] == false)
                    {
                        hit_truth_table[truth.second] = true;
                        truth_score_hits[truth.second] += score;
                    }
                    else
                    {
                        truth_score_hits[truth.second] += score + loss_per_false_alarm;
                    }
                }
            }

            hit_truth_table.assign(hit_truth_table.size(), false);

            final_dets.clear();
#ifdef ENABLE_ASSERTS
            double total_score = 0;
#endif
            // Now figure out which detections jointly maximize the loss and detection score sum.  We
            // need to take into account the fact that allowing a true detection in the output, while
            // initially reducing the loss, may allow us to increase the loss later with many duplicate
            // detections.
            for (unsigned long i = 0; i < dets.size() && final_dets.size() < max_num_dets; ++i)
            {
                if (overlaps_any_box(boxes_overlap, final_dets, dets[i].second))
                    continue;

                const std::pair<double,unsigned int> truth = find_best_match(truth_object_detections[idx], dets[i].second);

                const double truth_match = truth.first;
                if (truth_match > match_eps)
                {
                    if (truth_score_hits[truth.second] > loss_per_missed_target)
                    {
                        if (!hit_truth_table[truth.second])
                        {
                            hit_truth_table[truth.second] = true;
                            final_dets.push_back(dets[i].second);
#ifdef ENABLE_ASSERTS
                            total_score += dets[i].first;
#endif
                            loss -= loss_per_missed_target;
                        }
                        else
                        {
                            final_dets.push_back(dets[i].second);
#ifdef ENABLE_ASSERTS
                            total_score += dets[i].first;
#endif
                            loss += loss_per_false_alarm;
                        }
                    }
                }
                else if (!overlaps_ignore_box(idx,dets[i].second))
                {
                    // didn't hit anything
                    final_dets.push_back(dets[i].second);
#ifdef ENABLE_ASSERTS
                    total_score += dets[i].first;
#endif
                    loss += loss_per_false_alarm;
                }
            }

            psi.set_size(get_num_dimensions());
            psi = 0;
            for (unsigned long i = 0; i < final_dets.size(); ++i)
                scanner.get_feature_vector(scanner.get_full_object_detection(final_dets[i], current_solution), psi);

            psi(scanner.get_num_dimensions()) = -1.0*final_dets.size();
        }


        bool overlaps_ignore_box (const long idx, const dlib::rectangle& rect) const
        {
            for (unsigned long i = 0; i < ignore[idx].size(); ++i)
            {
                if (ignore_overlap_tester(ignore[idx][i], rect))
                    return true;
            }
            return false;
        }

        std::pair<double,unsigned int> find_best_match( const std::vector<full_object_detection>& boxes, const rectangle rect ) const
        {
            double match = 0;
            unsigned int best_idx = 0;
            for (unsigned long i = 0; i < boxes.size(); ++i)
            {

                const unsigned long area = rect.intersect(boxes[i].get_rect()).area();
                if (area != 0)
                {
                    const double new_match = area / static_cast<double>((rect + boxes[i].get_rect()).area());
                    if (new_match > match)
                    {
                        match = new_match;
                        best_idx = i;
                    }
                }
            }

            return std::make_pair(match,best_idx);
        }

        struct init_scanners_helper
        {
            init_scanners_helper (	array<image_scanner_type>& scanners_,
            										const image_array_type& images_  ) :	scanners(scanners_),
            															 	 	 	 	 	 	 	 	 	 	 	 	images(images_)
            {
            }

            array<image_scanner_type>& 	scanners;
            const image_array_type& 	images;

            void operator() (long i) const
            {
                scanners[i].load(images[i]);
            }
        };

        void initialize_scanners (const image_scanner_type& scanner, unsigned long num_threads)
        {
            scanners.set_max_size(images.size());
            scanners.set_size(images.size());

            for (unsigned long i = 0; i < scanners.size(); ++i)
                scanners[i].copy_configuration(scanner);

            // now load the images into all the scanners
            parallel_for(num_threads, 0, scanners.size(), init_scanners_helper(scanners, images));
        }


        test_box_overlap boxes_overlap;

        mutable array<image_scanner_type> scanners;

        const image_array_type& images;
        const std::vector<std::vector<full_object_detection> >& truth_object_detections;
        const std::vector<std::vector<rectangle> >& ignore;
        const test_box_overlap ignore_overlap_tester;

        unsigned long max_num_dets;
        double match_eps;
        double loss_per_false_alarm;
        double loss_per_missed_target;
    };
// ----------------------------------------------------------------------------------------
}

#endif // DLIB_STRUCTURAL_SVM_ObJECT_DETECTION_PROBLEM_Hh_


