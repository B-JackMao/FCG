NDK=/home/ferey/Android/android-ndk-r14b-linux-x86_64/android-ndk-r14b
PLATFORM=$NDK/platforms/android-21/arch-arm/
PREBUILT=$NDK/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64
CC=$PREBUILT/bin/arm-linux-androideabi-gcc
./configure --prefix=/home/ferey/adb/jpeg-9e/jni/dist --host=arm CC="$CC --sysroot=$PLATFORM"

