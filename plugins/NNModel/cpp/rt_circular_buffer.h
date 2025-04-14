/*
* adapted for SuperCollider from nn_tilde circular_buffer
* CircularBufferCtrl operates on an already allocated buffer
* so that it can be allocated via RTAlloc on the real-time memory
*/
#pragma once
#include <cstring>
#include "SC_InlineBinaryOp.h"

namespace NN {
template <class in_type, class out_type> class RingBufCtrl {
protected:
  out_type* _buffer;
  size_t _max_size;

  int _head = 0;
  int _tail = 0;
  bool _full = false;
public:
  RingBufCtrl(out_type* buf, size_t size): _buffer(buf), _max_size(size) {
  };

  out_type* getBuffer() const { return _buffer; }
  bool full() const { return _full; };
  bool empty() const { 
    return (!_full && _head == _tail);
  }
  size_t readable() const { 
    return empty() ? 0 : _head > _tail ? _head - _tail : _max_size - (_tail - _head); 
  }
  size_t writable() const {
    return _max_size - readable();
  }
  void reset() {;
    _head = _tail;
    _full = false;
  }

  void putRepeat(in_type val, int N) {
    put_impl(N, [&](int chunkSize, size_t written) {
      std::fill(&_buffer[_head], &_buffer[_head] + chunkSize, val);
    });
  }

  void put(const in_type *input_array, int N) {
    put_impl(N, [&](int chunkSize, size_t written) {
      memcpy(&_buffer[_head], &input_array[written], chunkSize * sizeof(in_type));
    });
  }

  void get(out_type *output_array, int N) {
    size_t read = 0;
    size_t bytesToRead = sc_min(readable(), N);

    while (read < bytesToRead) {
      int chunkSize = sc_min(bytesToRead - read, _max_size - _tail);
      memcpy(&output_array[read], &_buffer[_tail], chunkSize * sizeof(out_type));
      _tail = sc_mod(_tail + chunkSize, _max_size);
      read += chunkSize;
    }
    if (read < N)
      memset(&output_array[read], 0, sizeof(out_type) * (N-read));
    _full = false;
  };

private:
  // template for ar/kr put functions, writeChunk is given as lambda
  template <typename WriteFunc>
  void put_impl(int N, WriteFunc writeChunk) {
    size_t written = 0;

    // when overwriting old data, move _tail accordingly
    // otherwise readable() gets shorter
    if (N > writable())
      _tail = sc_mod(_tail + writable() - N, (int)_max_size);

    while (written < N) {
      int chunkSize = sc_min(N - written, _max_size - _head);
      writeChunk(chunkSize, written);
      _head = sc_mod(_head + chunkSize, _max_size);
      written += chunkSize;
    }

    // this works because N is always = blockSize, for both put and get
    if (_head == _tail) _full = true;
  }
};
}
