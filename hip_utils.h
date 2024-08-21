#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>

int checkHipError(const hipError_t err, const char* cmd)
{
  if(err) {
    printf("HIP error in command '%s'\n", cmd); \
    printf("Error message: %s\n", hipGetErrorString(err)); \
  }
  return err;
}

#define CHK_HIP_ERR(cmd) checkHipError(cmd, #cmd)
