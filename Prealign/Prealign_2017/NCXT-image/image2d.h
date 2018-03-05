#pragma once
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>
using namespace std;

template <class T> class image2d {
  private:
    unsigned int m_nx;
    unsigned int m_ny;

  public:
    ~image2d(){};
    vector<T> m_data;

    image2d() : m_data(vector<T>()), m_nx(0), m_ny(0){};
    image2d(int x, int y) : m_data(vector<T>(x * y, {0})), m_nx(x), m_ny(y){};
    image2d(int x, int y, T val) : m_data(vector<T>(x * y, val)), m_nx(x), m_ny(y){};

    void init(int x, int y, int z) { init(x, y, z, 0.0); }
    void init(int x, int y, int z, T val) {
        m_nx = x;
        m_ny = y;
        m_data = vector<T>(x * y, val);
    };

    void reset() { *this *= 0.0f; }

    T blerp(double x, double y) const {
        if (x < 0) return {0};
        if (y < 0) return {0};
        if (x > m_nx - 1) return {0};
        if (y > m_ny - 1) return {0};

        int xbase = (int)x;
        int ybase = (int)y;
        // Because the interpolation fraction is positive,
        // base integer is bounded by n-2
        if (xbase == m_nx - 1) xbase--;
        if (ybase == m_ny - 1) ybase--;

        double xFraction = x - xbase;
        double yFraction = y - ybase;
        assert(xFraction >= 0.0);
        assert(yFraction >= 0.0);

        double lowerLeft = (*this)(xbase, ybase);
        double lowerRight = (*this)(xbase + 1, ybase);
        double upperRight = (*this)(xbase + 1, ybase + 1);
        double upperLeft = (*this)(xbase, ybase + 1);

        double upperAverage = upperLeft + xFraction * (upperRight - upperLeft);
        double lowerAverage = lowerLeft + xFraction * (lowerRight - lowerLeft);
        return static_cast<T>(lowerAverage + yFraction * (upperAverage - lowerAverage));
    }

    T operator()(int i, int j) const {
        assert(i >= 0 && i < static_cast<int>(m_nx);
        assert(j >= 0 && j < static_cast<int>(m_ny);
        return m_data[m_nx * j + i];
    }
    T &operator()(int i, int j) {
        assert(i >= 0 && i < static_cast<int>m_nx));
        assert(j >= 0 && j < static_cast<int>m_ny));
        return m_data[m_nx * j + i];
    }
    T operator[](int i) const { return m_data[i]; }
    T &operator[](int i) { return m_data[i]; }

    // copy constructor
    image2d(const image2d &a) {
#ifdef DEBUG
        printf("copy constructor\n");
#endif // DEBUG
        m_nx = a.m_nx;
        m_ny = a.m_ny;
        m_data = std::vector<T>(a.m_data);
    }
    // copy assignment
    image2d &operator=(const image2d &a) {
#ifdef DEBUG
        printf("copy assignment\n");
#endif // DEBUG
        m_nx = a.m_nx;
        m_ny = a.m_ny;
        m_data = std::vector<T>(a.m_data);
        return *this;
    }
    // move constructor
    image2d(image2d &&a) : m_data{a.m_data}, m_nx{a.m_nx}, m_ny{a.m_ny} {
#ifdef DEBUG
        printf("move constructor\n");
#endif // DEBUG
        a.m_nx = 0;
        a.m_ny = 0;
        a.m_data.clear();
    }
    // move assignment
    image2d &operator=(image2d &&a) {
#ifdef DEBUG
        printf("move assignment\n");
#endif // DEBUG
        m_nx = a.m_nx;
        m_ny = a.m_ny;
        m_data = a.m_data;

        a.m_nx = 0;
        a.m_ny = 0;
        a.m_data.clear();

        return *this;
    }

    image2d &operator+=(const image2d &a) {
        assert(this->m_data.size() == a.m_data.size());
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] += a.m_data[i];
        }
        return *this;
    }
    image2d &operator-=(const image2d &a) {
        assert(this->m_data.size() == a.m_data.size());
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] -= a.m_data[i];
        }
        return *this;
    }
    image2d &operator/=(const image2d &a) {
        assert(this->m_data.size() == a.m_data.size());
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] /= a.m_data[i];
        }
        return *this;
    }

    image2d &operator*=(const image2d &a) {
        assert(this->m_data.size() == a.m_data.size());
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] *= a.m_data[i];
        }
        return *this;
    }

    image2d &operator+=(T a) {
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] += a;
        }
        return *this;
    }

    image2d &operator-=(T a) {
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] -= a;
        }
        return *this;
    }
    image2d &operator/=(T a) {
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] /= a;
        }
        return *this;
    }

    image2d &operator*=(T a) {
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] *= a;
        }
        return *this;
    }

    // Dimensions
    int nx() const { return static_cast<int>(m_nx); }
    int ny() const { return static_cast<int>(m_ny); }
    int size() const { return m_data.size(); }

    /* POINTERS */
    T *data() { return m_data.data(); }
    const T *data() const { return m_data.data(); }
    T *data_ind(int i) { return &m_data[i]; }
    const T *data_ind(int i) const { return &m_data[i]; }

    /* ITERATORS */
    auto begin() { return m_data.begin(); }
    auto end() { return m_data.end(); }

    template <typename F> void transform(F func) {
        std::transform(m_data.begin(), m_data.end(), m_data.begin(), func);
    }
    template <typename F> void apply(F func) { std::for_each(m_data.begin(), m_data.end(), func); }
};

template <typename F, class T> void apply(image2d<T> a, F func) {
    std::for_each(a.begin(), a.end(), func);
}

template <class T> image2d<T> operator+(image2d<T> a, const image2d<T> &b) { return a += b; }
template <class T> image2d<T> operator-(image2d<T> a, const image2d<T> &b) { return a -= b; }
template <class T> image2d<T> operator/(image2d<T> a, const image2d<T> &b) { return a /= b; }
template <class T> image2d<T> operator*(image2d<T> a, const image2d<T> &b) { return a *= b; }
template <class T> image2d<T> operator+(image2d<T> a, T b) { return a += b; }
template <class T> image2d<T> operator-(image2d<T> a, T b) { return a -= b; }
template <class T> image2d<T> operator/(image2d<T> a, T b) { return a /= b; }
template <class T> image2d<T> operator*(image2d<T> a, T b) { return a *= b; }
