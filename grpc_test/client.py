#! /usr/bin/env python
# -*- coding: utf-8 -*-
import grpc
from msg import msg_pb2, msg_pb2_grpc

_HOST = '163.143.0.101'
_PORT = '6003'

def run():
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)
    client = msg_pb2_grpc.MsgServiceStub(channel=conn)
    response = client.GetMsg(msg_pb2.RequestData(text='hello,world!'))
    print("received: " + response.text)

if __name__ == '__main__':
    run()