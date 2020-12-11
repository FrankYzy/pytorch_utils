# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author         :  Frank
@Version        :  
------------------------------------
@File           :  loggers.py
@Description    :  
@CreateTime     :  2020/5/21 16:10
------------------------------------
@ModifyTime     :  
"""
import sys
import os
import errno


class Logger(object):
    def __init__(self, filepath):
        self.console = sys.stdout
        self.log = None
        if filepath is not None:
            # 判断文件所在目录是否存在，如果不存在则创建路径上的目录，再创建文件
            if not os.path.exists(os.path.dirname(filepath)):
                try:
                    os.makedirs(os.path.dirname(filepath))
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            self.log = open(filepath, 'w+')

    def write(self, msg):
        self.console.write(msg)
        if self.log is not None:
            self.log.write(msg)

    def flush(self):
        pass
