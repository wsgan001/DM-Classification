CC          := g++
EXE         := main.py
PKG_LIST    := requirements.txt
OBJ         := main.o
INCLUDE_DIR := ./include/
LIB_DIR     := ./lib/
LIB_DT_DIR  := ./dtree/
LIB_IF_DIR  := ./cwrapper/
LIB_DT      := DecisionTree
LIB_IF      := CWrapper
LIB_DT_SRC  := DecisionTree.cpp
LIB_IF_SRC  := CWrapper.cpp
SRC         := main.cpp
CC_FLAG     := -L${LIB_DIR} -Wl,-rpath,${LIB_DIR} -l${LIB_IF} -l${LIB_DT}
CC_INCLUDE  := -I${INCLUDE_DIR}
EXE_ARG     := 
LIB_DT_SRC_PATH  := $(foreach src,${LIB_DT_SRC},${LIB_DT_DIR}${src})
LIB_IF_SRC_PATH := $(foreach src,${LIB_IF_SRC},${LIB_IF_DIR}${src})


all: clean ${LIB_DIR}lib${LIB_IF}.so ${LIB_DIR}lib${LIB_DT}.so

${LIB_DIR}lib${LIB_DT}.so:
	${CC} ${LIB_DT_SRC_PATH} -fPIC -shared -o ${LIB_DIR}lib${LIB_DT}.so ${CC_INCLUDE}

${LIB_DIR}lib${LIB_IF}.so: ${LIB_DIR}lib${LIB_DT}.so
	${CC} ${LIB_IF_SRC_PATH} -fPIC -shared -o ${LIB_DIR}lib${LIB_IF}.so ${CC_INCLUDE} \
        -l${LIB_DT} -L${LIB_DIR} -Wl,-rpath,${LIB_DIR}

package: ${PKG_LIST}
	pip install -r ${PKG_LIST}

run: all ${EXE} 
	python3 ${EXE}

clean:
	rm -rf ${LIB_DIR}*.so *.o
