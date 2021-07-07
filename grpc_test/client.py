#! /usr/bin/env python
# -*- coding: utf-8 -*-
import grpc
from msg import msg_pb2, msg_pb2_grpc

_HOST = '163.143.0.120'
_PORT = '6003'

def run():
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)
    client = msg_pb2_grpc.msgServiceStub(channel=conn)
    response = client.SimpleFun(msg_pb2.RequestData(text='hello,world!'))
    print("received: " + response.text)

if __name__ == '__main__':
    run()