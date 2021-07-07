#! /usr/bin/env python
# -*- coding: utf-8 -*-
import grpc
import time
from concurrent import futures
from msg import msg_pb2, msg_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '163.143.0.101'
_PORT = '6003'

# export GRPC_VERBOSITY = DEBUG

class servicer(msg_pb2_grpc.MsgServiceServicer):

    def SimpleFun(self, request, context):
        str = request.text
        print("received: " + str)
        return msg_pb2.ResponseData(text=('hello,gRPC'))


def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    msg_pb2_grpc.add_MsgServiceServicer_to_server(servicer(), grpcServer)
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)
    grpcServer.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)


if __name__ == '__main__':
    serve()