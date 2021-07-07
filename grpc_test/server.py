#! /usr/bin/env python
# -*- coding: utf-8 -*-
import grpc
import time
from concurrent import futures
from msg import msg_pb2, msg_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '163.143.0.101'
_PORT = '6003'


class servicer(stream_pb2_grpc.StreamServiceServicer):

    def SimpleFun(self, request, context):
        str = request.text
        print("received: " + str)
        return stream_pb2.ResponseData(text=('hello,gRPC'))


def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    stream_pb2_grpc.add_StreamServiceServicer_to_server(servicer(), grpcServer)
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)
    grpcServer.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)


if __name__ == '__main__':
    serve()