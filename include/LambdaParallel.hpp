#pragma once
template<class _Function>
class LambdaBody : public cv::ParallelLoopBody {
public:
	typedef _Function function_type;
	typedef const _Function const_function_type;

	inline LambdaBody(const_function_type& body) :
		_body(body)
	{}

	inline void operator() (const cv::Range & range) const
	{
		this->_body(range);
	}
private:
	const_function_type _body;
};

template<class _Function>
inline void parallel_for(const cv::Range& range, const _Function& body, const double& nstride = -1.)
{
	cv::parallel_for_(range, LambdaBody<_Function>(body), nstride);
}