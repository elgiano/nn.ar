#include "SC_Reply.h"

#include <boost/asio/ip/address.hpp>

enum Protocol { kUDP, kTCP };

struct ReplyAddress {
    boost::asio::ip::address mAddress;
    enum Protocol mProtocol;
    int mPort;
    int mSocket;

    ReplyFunc mReplyFunc;
    void* mReplyData;
};

inline void SendReply(struct ReplyAddress* inReplyAddr, char* inBuf, int inSize) {
    (inReplyAddr->mReplyFunc)(inReplyAddr, inBuf, inSize);
};
