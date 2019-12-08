NVCC		= nvcc
CC		= g++
CU_FLAGS	= -O3 -arch=sm_52
CU1_FLAGS	= -O0 -Xcicc -O0 -Xptxas -O0
CC_FLAGS	= -O3 -m64 -Wall

CU_SOURCES	= reversesetmapping.cu


CU_OBJECTS	= $(CU_SOURCES:%.cu=%.o)
CU_PTX		= $(CU_SOURCES:%.cu=%.ptx)
CC_OBJECTS	= $(CC_SOURCES:%.cc=%.o)

%.o:		%.cu
		$(NVCC) $(CU_FLAGS) -dc $< -o $@

%.ptx:		%.cu
		$(NVCC) $(CU_FLAGS) --ptx $< -o $@

reversesetmapping:	$(CU_OBJECTS) $(CC_OBJECTS)
		$(NVCC) -arch=sm_52 $^ -o $@

ptx:		$(CU_PTX) 

clean:
		rm -f *.o reversesetmapping
