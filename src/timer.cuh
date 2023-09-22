#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <string>

namespace cpu {
	class timer {
	private:
		std::chrono::time_point<std::chrono::steady_clock> m_start, m_end;
		std::chrono::milliseconds m_duration;
		std::string m_label;
		bool m_isTiming;
	public:
		timer() : m_label(""), m_isTiming(false) {}
		timer(const std::string &_label) : m_label(_label), m_isTiming(false) {}

		std::string label() const {
			return m_label;
		}
		void label(const std::string &_label) {
			m_label = _label;
		}
		void start() {
			m_isTiming = true;
			m_start = std::chrono::high_resolution_clock::now();
		}
		long long end() {
			m_isTiming = false;
			m_end = std::chrono::high_resolution_clock::now();
			m_duration = std::chrono::duration_cast<std::chrono::milliseconds>(m_end - m_start);
			return m_duration.count();
		}
		long long get() const {
			return m_duration.count();
		}
	};
}

namespace gpu {
	class timer {
	private:
		cudaEvent_t m_start, m_end;
		long long m_duration;
		std::string m_label;
		bool m_isTiming;
	public:
		timer() : m_label(""), m_isTiming(false) {}
		timer(const std::string &_label) : m_label(_label), m_isTiming(false) {}

		std::string label() {
			return m_label;
		}
		void label(const std::string &_label) {
			m_label = _label;
		}
		void start() {
			m_isTiming = true;
			cudaEventCreate(&m_start);
			cudaEventCreate(&m_end);
			cudaEventRecord(m_start, 0);
		}
		long long end() {
			m_isTiming = false;
			float elapsedTime;
			cudaEventRecord(m_end, 0);
			cudaEventSynchronize(m_end);
			cudaEventElapsedTime(&elapsedTime, m_start, m_end);
			cudaEventDestroy(m_start);
			cudaEventDestroy(m_end);
			m_duration = (long long)elapsedTime;
			return m_duration;
		}
		long long get() {
			return m_duration;
		}
	};
}