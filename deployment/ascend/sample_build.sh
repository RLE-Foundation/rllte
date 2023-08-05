#!/bin/bash

model_name="MyFirstApp_build"

cd ${APP_SOURCE_PATH}/data

python3 ../script/transferPic.py

if [ -d ${APP_SOURCE_PATH}/build/intermediates/host ];then
	rm -rf ${APP_SOURCE_PATH}/build/intermediates/host
fi

mkdir -p ${APP_SOURCE_PATH}/build/intermediates/host
cd ${APP_SOURCE_PATH}/build/intermediates/host

cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE

make

if [ $? == 0 ];then
	echo "make for app ${model_name} Successfully"
	exit 0
else
	echo "make for app ${model_name} failed"
	exit 1
fi