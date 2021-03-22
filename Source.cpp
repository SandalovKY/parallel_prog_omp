#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <omp.h>


std::vector<double> createGaussianFilter(double sigma, size_t N) {
	if (N % 2 == 0) throw std::runtime_error("Invalid Gaussian kernel size");
	std::vector<double> imgFilter(N*N);
	int64_t  n = N / 2;
	double sum{ 0 };
	for (int64_t i = -n; i <= n; ++i) {
		for (int64_t j = -n; j <= n; ++j) {
			double r = static_cast<double>(i * i + j * j);
			double s{ 2.0 * sigma * sigma };

			imgFilter[(n + i) * N + (n + j)] = static_cast<double> (exp(-(r / s)) / (3.14 * s));
			sum += imgFilter[(n + i) * N + (n + j)];
		}
	}
	for (int64_t i = 0; i < N; ++i) {
		for (int64_t j = 0; j < N; ++j) {
			imgFilter[i * N + j] /= sum;
		}
	}
	return imgFilter;
}

double applyKernel(const std::vector<double>& srcImg, const std::vector<double>& kernel, int64_t X, int64_t Y, size_t N, size_t M, size_t K) {

	double resValue{ 0 };

	int64_t width = K / 2;

	//#pragma omp simd reduction (+:resValue)
	for (int64_t i = -width; i <= width; ++i) {
		for (int64_t j = -width; j <= width; ++j) {
			size_t curX = std::abs(X + i);
			if (curX >= N) curX = X - i;
			size_t curY = std::abs(Y + j);
			if (curY >= M) curY = Y - j;
			resValue += srcImg[(curX)*M + (curY)] * kernel[(width + i) * K + (width + j)];
		}
	}
	return resValue;
}

std::vector<double> linearFiltering(const std::vector<double>& srcImg, const std::vector<double>& kernel, int64_t N, int64_t M, int64_t K) {
	std::vector<double> resImg(N * M);

	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < K; ++i) {
		omp_set_num_threads(4);

#pragma omp parallel for firstprivate(N, M, K)
		for (int j = i; j < N; j += K) {
			for (int f = 0; f < M; ++f) {
				resImg[j * M + f] = applyKernel(srcImg, kernel, j, f, N, M, K);
			}
		}
	}
	auto end = std::chrono::system_clock::now();

	auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << '\t' << elapsedTime.count() << std::endl;
	return resImg;
}

int main() {
	constexpr uint32_t kernelSize = 3;
	std::string imPath = "C:\\Users\\sanda\\Desktop\\kurisu.jpg";
	cv::Mat image = cv::imread(imPath, cv::IMREAD_GRAYSCALE);

	if (image.empty())
	{
		std::cout << "Could not read the image: " << imPath << std::endl;
		return 1;
	}
	cv::imshow("Window", image);
	int k = cv::waitKey(0);

	image.convertTo(image, CV_64FC1);

	auto width = image.cols;
	auto height = image.rows;
	std::vector<double> imgStr (image.begin<double>(), image.end<double>());
	
	auto gaussian = createGaussianFilter(1, kernelSize);
	std::vector<double> resImage = linearFiltering(imgStr, gaussian, height, width, kernelSize);

	cv::Mat matRes = cv::Mat(height, width, CV_64FC1, resImage.data());
	matRes.convertTo(matRes, CV_8U);
	cv::imshow("Window1", matRes);
	k = cv::waitKey(0);
}