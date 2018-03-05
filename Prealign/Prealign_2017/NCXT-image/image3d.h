#pragma once
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>
using namespace std;

template <class T> class image3d {
  public:
    vector<T> m_data;

  private:
    size_t m_nx;
    size_t m_ny;
    size_t m_nz;

  public:
    ~image3d(){};

    image3d() : m_data(vector<T>()), m_nx(0), m_ny(0), m_nz(0){};

    image3d(int x, int y, int z) : m_data(vector<T>(x * y * z, {0})), m_nx(x), m_ny(y), m_nz(z){};

    image3d(int x, int y, int z, T val)
        : m_data(vector<T>(x * y * z, val)), m_nx(x), m_ny(y), m_nz(z){};

    T operator()(int i, int j, int k) const {
        assert(i > 0 && i < m_nx);
        assert(j > 0 && j < m_ny);
        assert(k > 0 && k < m_nz);
        return m_data[m_nx * (m_ny * k + j) + i];
    }
    T &operator()(int i, int j, int k) {
        assert(i > 0 && i < m_nx);
        assert(j > 0 && j < m_ny);
        assert(k > 0 && k < m_nz);
        return m_data[m_nx * (m_ny * k + j) + i];
    }
    T operator[](int i) const { return m_data[i]; }
    T &operator[](int i) { return m_data[i]; }

    // copy constructor
    image3d(const image3d &a) {
#ifdef DEBUG
        printf("copy constructor\n");
#endif // DEBUG
        m_nx = a.m_nx;
        m_ny = a.m_ny;
        m_nz = a.m_nz;
        m_data = std::vector<T>(a.m_data);
    }
    // copy assignment
    image3d &operator=(const image3d &a) {
#ifdef DEBUG
        printf("copy assignment\n");
#endif // DEBUG
        m_nx = a.m_nx;
        m_ny = a.m_ny;
        m_nz = a.m_nz;
        m_data = std::vector<T>(a.m_data);
        return *this;
    }
    // move constructor
    image3d(image3d &&a) : m_data{a.m_data}, m_nx{a.m_nx}, m_ny{a.m_ny}, m_nz{a.m_nz} {
#ifdef DEBUG
        printf("move constructor\n");
#endif // DEBUG
        a.m_nx = 0;
        a.m_ny = 0;
        a.m_nz = 0;
        a.m_data.clear();
    }
    // move assignment
    image3d &operator=(image3d &&a) {
#ifdef DEBUG
        printf("move assignment\n");
#endif // DEBUG
        m_nx = a.m_nx;
        m_ny = a.m_ny;
        m_nz = a.m_nz;
        m_data = a.m_data;

        a.m_nx = 0;
        a.m_ny = 0;
        a.m_nz = 0;
        a.m_data.clear();

        return *this;
    }

    image3d &operator+=(const image3d &a) {
        assert(this->m_data.size() == a.m_data.size());
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] += a.m_data[i];
        }
        return *this;
    }
    image3d &operator-=(const image3d &a) {
        assert(this->m_data.size() == a.m_data.size());
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] -= a.m_data[i];
        }
        return *this;
    }
    image3d &operator/=(const image3d &a) {
        assert(this->m_data.size() == a.m_data.size());
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] /= a.m_data[i];
        }
        return *this;
    }

    image3d &operator*=(const image3d &a) {
        assert(this->m_data.size() == a.m_data.size());
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] *= a.m_data[i];
        }
        return *this;
    }

    image3d &operator+=(T a) {
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] += a;
        }
        return *this;
    }

    image3d &operator-=(T a) {
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] -= a;
        }
        return *this;
    }
    image3d &operator/=(T a) {
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] /= a;
        }
        return *this;
    }

    image3d &operator*=(T a) {
        for (size_t i = 0; i < this->m_data.size(); ++i) {
            this->m_data[i] *= a;
        }
        return *this;
    }

    // Dimensions
    int nx() const { return static_cast<int>(m_nx); }
    int ny() const { return static_cast<int>(m_ny); }
    int nz() const { return static_cast<int>(m_nz); }
    size_t size() const { return m_data.size(); }

    /* POINTERS */
    T *data() { return m_data.data(); }
    const T *data() const { return m_data.data(); }
    T *data_ind(int i) { return &m_data[i]; }
    const T *data_ind(int i) const { return &m_data[i]; }
    T *data_slice(int i) { return &m_data[m_nx * m_ny * i]; }
    const T *data_slice(int i) const { return &m_data[m_nx * m_ny * i]; }

    /* ITERATORS */
    auto begin() { return m_data.begin(); }
    auto end() { return m_data.end(); }

    auto it_data(int i) { return std::next(m_data.begin(), i); }
    auto it_slice(int i) { return std::next(m_data.begin(), m_nx * m_ny * i); }

    template <typename F> void transform(F func) {
        std::transform(m_data.begin(), m_data.end(), m_data.begin(), func);
    }
    template <typename F> void apply(F func) { std::for_each(m_data.begin(), m_data.end(), func); }
};

template <typename F, class T> void apply(image3d<T> a, F func) {
    std::for_each(a.begin(), a.end(), func);
}

template <class T> image3d<T> operator+(image3d<T> a, const image3d<T> &b) { return a += b; }
template <class T> image3d<T> operator-(image3d<T> a, const image3d<T> &b) { return a -= b; }
template <class T> image3d<T> operator/(image3d<T> a, const image3d<T> &b) { return a /= b; }
template <class T> image3d<T> operator*(image3d<T> a, const image3d<T> &b) { return a *= b; }
template <class T> image3d<T> operator+(image3d<T> a, T b) { return a += b; }
template <class T> image3d<T> operator-(image3d<T> a, T b) { return a -= b; }
template <class T> image3d<T> operator/(image3d<T> a, T b) { return a /= b; }
template <class T> image3d<T> operator*(image3d<T> a, T b) { return a *= b; }

template <class T> bool operator==(const image3d<T> &lhs, const image3d<T> &rhs) {
    assert(lhs.m_data.size() == rhs.m_data.size());
    for (size_t i = 0; i < lhs.m_data.size(); ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}
template <class T> bool operator!=(const image3d<T> &lhs, const image3d<T> &rhs) {
    return !(lhs == rhs);
}
template <class T> bool operator<(const image3d<T> &lhs, const image3d<T> &rhs) {
    assert(lhs.m_data.size() == rhs.m_data.size());
    for (size_t i = 0; i < lhs.m_data.size(); ++i) {
        if (lhs[i] > rhs[i]) return false;
    }
    return true;
}
template <class T> bool operator>(const image3d<T> &lhs, const image3d<T> &rhs) {
    return rhs < lhs;
}
