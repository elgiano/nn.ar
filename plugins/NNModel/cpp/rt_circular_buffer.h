/*
* adapted for SuperCollider from nn_tilde circular_buffer
* CircularBufferCtrl operates on an already allocated buffer
* so that it can be allocated via RTAlloc on the real-time memory
*/
#pragma once
#include "SC_World.h"
#include <cstring>
#include <memory>
#include <iostream>
#include "SC_InlineBinaryOp.h"

namespace NN {
template <class in_type, class out_type> class RingBufCtrl {
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

  void put(const in_type *input_array, int N) {
    size_t written = 0;

    while (written < N) {
      int chunkSize = sc_min(N - written, _max_size - _head);
      memcpy(&_buffer[_head], &input_array[written], chunkSize * sizeof(out_type));
      _head = sc_mod(_head + chunkSize, _max_size);
      written += chunkSize;
    }

    if (_head == _tail) _full = true;
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
    if (bytesToRead < N)
      memset(&output_array[bytesToRead], 0, sizeof(out_type) * (N-bytesToRead));
    _full = false;
  };

  void reset() {;
    _head = _tail;
    _full = false;
  }

protected:
  out_type* _buffer;
  size_t _max_size;

  int _head = 0;
  int _tail = 0;
  bool _full = false;
};
}
