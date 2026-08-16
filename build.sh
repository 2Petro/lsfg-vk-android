rm -rf build-android

cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=/root/ndk/r29/build/cmake/android.toolchain.cmake \
  -DANDROID_NDK=/root/ndk/r29 \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-35 \
  -DLSFGVK_ANDROID_WINE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-android -j$(nproc)
