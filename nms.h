#ifndef NMS_H
#define NMS_H

/**
 * @brief nms
 * Non maximum suppression
 * @param srcRects
 * @param dstRects
 * @param thresh
 * @param neighbors
 */
inline void nms(std::vector<cv::Rect>&  srcRects,
                std::vector<cv::Rect>&  dstRects,
                float   thresh,
                int     neighbors = 0)
{
    dstRects.clear();

    const size_t size = srcRects.size();
    if (!size)
        return;

    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    std::multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.emplace(srcRects[i].br().y, i);
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];

        int neigborsCount = 0;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

            float intArea = static_cast<float>((rect1 & rect2).area());
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors)
            dstRects.push_back(rect1);
    }
}

/**
 * @brief nms2
 * Non maximum suppression with detection scores
 * @param srcRects
 * @param scores
 * @param dstRects
 * @param thresh
 * @param neighbors
 */
inline void nms2(const std::vector<cv::Rect>& srcRects,
                 const std::vector<float>& scores,
                 std::vector<cv::Rect>& dstRects,
                 float thresh,
                 int neighbors = 0,
                 float minScoresSum = 0.f)
{
    dstRects.clear();

    const size_t size = srcRects.size();
    if (!size)
        return;

    assert(srcRects.size() == scores.size());

    // Sort the bounding boxes by the detection score
    std::multimap<float, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.emplace(scores[i], i);
    }

    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];

        int neigborsCount = 0;
        float scoresSum = lastElem->first;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];

            float intArea = static_cast<float>((rect1 & rect2).area());
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;

            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                scoresSum += pos->first;
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors && scoresSum >= minScoresSum)
            dstRects.push_back(rect1);
    }
}

///
enum class Methods
{
	ClassicNMS,
	LinearNMS,
	GaussNMS
};

/**
 * @brief nms2
 * Non maximum suppression with detection scores
 * @param srcRects
 * @param scores
 * @param dstRects
 * @param thresh
 */
inline void soft_nms(const std::vector<cv::Rect>& srcRects,
                     const std::vector<float>& scores,
                     std::vector<cv::Rect>& dstRects,
                     std::vector<float>& resScores,
                     float iou_thresh,
                     float score_thresh,
                     Methods method,
                     float sigma = 0.5f)
{
	dstRects.clear();

	const size_t size = srcRects.size();
	if (!size)
		return;

	assert(srcRects.size() == scores.size());

	// Sort the bounding boxes by the detection score
	std::multimap<float, size_t> idxs;
	for (size_t i = 0; i < size; ++i)
	{
		if (scores[i] >= score_thresh)
			idxs.emplace(scores[i], i);
	}

	if (dstRects.capacity() < idxs.size())
	{
		dstRects.reserve(idxs.size());
		resScores.reserve(idxs.size());
	}

	// keep looping while some indexes still remain in the indexes list
	while (idxs.size() > 0)
	{
		// grab the last rectangle
		auto lastElem = --std::end(idxs);
		const cv::Rect& rect1 = srcRects[lastElem->second];

		if (lastElem->first >= score_thresh)
		{
			dstRects.push_back(rect1);
			resScores.push_back(lastElem->first);
		}
		else
		{
			break;
		}
		idxs.erase(lastElem);

		for (auto pos = std::begin(idxs); pos != std::end(idxs); )
		{
			// grab the current rectangle
			const cv::Rect& rect2 = srcRects[pos->second];

			float intArea = static_cast<float>((rect1 & rect2).area());
			float unionArea = rect1.area() + rect2.area() - intArea;
			float overlap = intArea / unionArea;

			// if there is sufficient overlap, suppress the current bounding box
			if (overlap > iou_thresh)
			{
				float weight = 1.f;
				switch (method)
				{
				case Methods::ClassicNMS:
					weight = 0;
					break;
				case Methods::LinearNMS:
					weight = 1.f - overlap;
					break;
				case Methods::GaussNMS:
					weight = exp(-(overlap * overlap) / sigma);
					break;
				}

				float newScore = pos->first * weight;
				if (newScore < score_thresh)
				{
					pos = idxs.erase(pos);
				}
				else
				{
					//auto n = idxs.extract(pos);
					//n.key() = newScore;
					//idxs.insert(std::move(n));
					idxs.insert(std::multimap<float, size_t>::value_type(newScore, pos->second));
					idxs.erase(pos);
					++pos;
				}
			}
			else
			{
				++pos;
			}
		}
	}
}

#endif
