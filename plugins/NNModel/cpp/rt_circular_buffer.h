/*
* adapted for SuperCollider from nn_tilde circular_buffer
* CircularBufferCtrl operates on an already allocated buffer
* so that it can be allocated via RTAlloc on the real-time memory
*/
#pragma once
#include "SC_World.h"
#include <memory>
#include <iostream>

namespace NN {
template <class in_type, class out_type> class RingBufCtrl {
public:
  RingBufCtrl(out_type* buf, size_t size): _buffer(buf), _max_size(size) {
  };
  out_type* getBuffer() { return _buffer; }
  bool empty() { 
    return (!_full && _head == _tail);
  }
  bool full() { return _full; };
  void put(const in_type *input_array, int N) {
    if (!_max_size)
      return;

    while (N--) {
      _buffer[_head] = out_type(*(input_array++));
      _head = (_head + 1) % _max_size;
      if (_full)
        _tail = (_tail + 1) % _max_size;
      _full = _head == _tail;
    }
  }

  void get(out_type *output_array, int N) {
    if (!_max_size)
      return;

    while (N--) {
      if (empty()) {
        *(output_array++) = out_type();
      } else {
        *(output_array++) = _buffer[_tail];
        _tail = (_tail + 1) % _max_size;
        _full = false;
      }
    }
  };

  void reset() {;
    _head = _tail;
    _count = 0;
    _full = false;
  }

protected:
  out_type* _buffer;
  size_t _max_size;

  int _head = 0;
  int _tail = 0;
  int _count = 0;
  bool _full = false;
};
}
