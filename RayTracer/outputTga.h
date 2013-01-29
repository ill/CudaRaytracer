#ifndef OUTPUT_TGA_H__
#define OUTPUT_TGA_H__

#include <glm/glm.hpp>
#include <cstdio>
#include <stdint.h>

/**
 * Outputs some color data to TGA
 * @param color The color buffer
 * @param resolution The width and height dimensions of the color buffer
 * @param fileName the output filename
 */
inline void tgaOut(const uint32_t * color, const glm::uvec2& resolution, const char * fileName) {
   //Bob Somers's output TGA code, just a little bit modified by me

   FILE *fp = fopen(fileName, "wb");          //IT HAS TO BE wb not w so FRIGGIN WINDOWS doesn't think I'm trying to do a newline and prints 2 chars instead of 1 if the byte value is 10
   if (fp == NULL)
   {
      perror("ERROR: Image::WriteTga() failed to open file for writing!\n");
      exit(EXIT_FAILURE);
   }

   // write 24-bit uncompressed targa header
   // thanks to Paul Bourke (http://local.wasp.uwa.edu.au/~pbourke/dataformats/tga/)
   putc(0, fp);
   putc(0, fp);

   putc(2, fp); // type is uncompressed RGB

   putc(0, fp);
   putc(0, fp);
   putc(0, fp);
   putc(0, fp);
   putc(0, fp);

   putc(0, fp); // x origin, low byte
   putc(0, fp); // x origin, high byte

   putc(0, fp); // y origin, low byte
   putc(0, fp); // y origin, high byte

   putc(resolution.x & 0xff, fp); // width, low byte
   putc((resolution.x & 0xff00) >> 8, fp); // width, high byte

   putc(resolution.y & 0xff, fp); // height, low byte
   putc((resolution.y & 0xff00) >> 8, fp); // height, high byte

   putc(24, fp); // 24-bit color depth

   putc(0, fp);

   // write the raw pixel data in groups of 3 bytes (BGR order)
   for (unsigned int y = 0; y < resolution.y; y++) {
      for (unsigned int x = 0; x < resolution.x; x++) {
         const uint32_t& pixel = color[x + resolution.x * y];

         unsigned char rbyte, gbyte, bbyte;

         /*glm::mediump_float r = (pixel.x > 1.0f) ? 1.0f : pixel.x;
         glm::mediump_float g = (pixel.y > 1.0f) ? 1.0f : pixel.y;
         glm::mediump_float b = (pixel.z > 1.0f) ? 1.0f : pixel.z;*/
         rbyte = (unsigned char)((pixel & 0xFF000000) >> 24);
         gbyte = (unsigned char)((pixel & 0x00FF0000) >> 16);
         bbyte = (unsigned char)((pixel & 0x0000FF00) >> 8);

         putc(bbyte, fp);
         putc(gbyte, fp);
         putc(rbyte, fp);
      }
   }

   fclose(fp);
}

#endif
