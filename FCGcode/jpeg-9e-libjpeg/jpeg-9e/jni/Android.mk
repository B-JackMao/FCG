LOCAL_PATH := $(my-dir)

include $(CLEAR_VARS)
# From autoconf-generated Makefile
LOCAL_MODULE := libjpeg
LOCAL_ARM_MODE=arm
LOCAL_SRC_FILES := \
		   jaricom.c \
		   jcapimin.c \
		   jcapistd.c \
		   jcarith.c \
		   jccoefct.c \
		   jccolor.c \
		   jcdctmgr.c \
		   jchuff.c \
		   jcinit.c \
		   jcmainct.c \
		   jcmarker.c \
		   jcmaster.c \
		   jcomapi.c \
		   jcparam.c \
		   jcprepct.c \
		   jcsample.c \
		   jctrans.c \
		   jdapimin.c \
		   jdapistd.c \
		   jdarith.c \
		   jdatadst.c \
		   jdatasrc.c \
		   jdcoefct.c \
		   jdcolor.c \
		   jddctmgr.c \
		   jdhuff.c \
		   jdinput.c \
		   jdmainct.c \
		   jdmarker.c \
		   jdmaster.c \
		   jdmerge.c \
		   jdpostct.c \
		   jdsample.c \
		   jdtrans.c \
		   jerror.c \
		   jfdctflt.c \
		   jfdctfst.c \
		   jfdctint.c \
		   jidctflt.c \
		   jidctfst.c \
		   jidctint.c \
		   jquant1.c \
		   jquant2.c \
		   jutils.c \
		   jmemmgr.c \
		   jmemnobs.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)
LOCAL_CFLAGS :=-O3 -fstrict-aliasing -fprefetch-loop-arrays  -DANDROID \
        -DANDROID_TILE_BASED_DECODE -DENABLE_ANDROID_NULL_CONVERT

include $(BUILD_SHARED_LIBRARY)
