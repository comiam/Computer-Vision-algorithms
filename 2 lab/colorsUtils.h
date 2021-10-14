#ifndef COLORSUTILS_H
#define COLORSUTILS_H

#include <cmath>

struct RgbColor
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

struct HsvColor
{
    unsigned short h;
    unsigned char s;
    unsigned char v;
};

struct XyzColor
{
    double x;
    double y;
    double z;
};

struct LabColor
{
    double l;
    double a;
    double b;
};

inline HsvColor RgbToHsv(const RgbColor &rgb)
{
    HsvColor hsv;
    unsigned char rgbMin, rgbMax;

    rgbMin = rgb.r < rgb.g ? (rgb.r < rgb.b ? rgb.r : rgb.b) : (rgb.g < rgb.b ? rgb.g : rgb.b);
    rgbMax = rgb.r > rgb.g ? (rgb.r > rgb.b ? rgb.r : rgb.b) : (rgb.g > rgb.b ? rgb.g : rgb.b);

    hsv.v = rgbMax;
    if (hsv.v == 0)
    {
        hsv.h = 0;
        hsv.s = 0;
        return hsv;
    }

    hsv.s = 255 * long(rgbMax - rgbMin) / hsv.v;
    if (hsv.s == 0)
    {
        hsv.h = 0;
        return hsv;
    }

    int delta = rgbMax - rgbMin;
    if (rgbMax == rgb.r and rgb.g >= rgb.b)
        hsv.h = 60 * (rgb.g - rgb.b) / (delta);
    else if (rgbMax == rgb.r and rgb.g < rgb.b)
        hsv.h = 360 + 60 * (rgb.g - rgb.b) / delta;
    else if (rgbMax == rgb.g)
        hsv.h = 120 + 60 * (rgb.b - rgb.r) / delta;
    else
        hsv.h = 240 + 60 * (rgb.r - rgb.g) / delta;

    return hsv;

}

inline XyzColor rgbToXyz(const RgbColor &rgb)
{
    XyzColor xyz;
    double var_r = rgb.r / 255.0;
    double var_g = rgb.g / 255.0;
    double var_b = rgb.b / 255.0;

    if (var_r > 0.04045)
        var_r = pow((var_r + 0.055) / 1.055, 2.4);
    else
        var_r = var_r / 12.92;
    if (var_g > 0.04045)
        var_g = pow(((var_g + 0.055) / 1.055), 2.4);
    else
        var_g = var_g / 12.92;
    if (var_b > 0.04045)
        var_b = pow(((var_b + 0.055) / 1.055), 2.4);
    else
        var_b = var_b / 12.92;

    xyz.x = var_r * 0.4124 + var_g * 0.3576 + var_b * 0.1805;
    xyz.y = var_r * 0.2126 + var_g * 0.7152 + var_b * 0.0722;
    xyz.z = var_r * 0.0193 + var_g * 0.1192 + var_b * 0.9505;

    xyz.x *= 100;
    xyz.y *= 100;
    xyz.z *= 100;

    return xyz;
}

inline LabColor xyzToLab(const XyzColor &xyz)
{
    LabColor labColor;

    double ref_x = 95.04;
    double ref_y = 100.000;
    double ref_z = 108.88;

    double var_x = xyz.x / ref_x;
    double var_y = xyz.y / ref_y;
    double var_z = xyz.z / ref_z;

    if (var_x > 0.008856)
        var_x = pow(var_x, (0.333));
    else
        var_x = (7.787 * var_x) + (0.097);
    if (var_y > 0.008856)
        var_y = pow(var_y, (0.333));
    else
        var_y = (7.787 * var_y) + (0.097);
    if (var_z > 0.008856)
        var_z = pow(var_z, (0.333));
    else
        var_z = (7.787 * var_z) + (0.097);

    labColor.l = 116 * var_y - 16;
    labColor.a = 500 * (var_x - var_y);
    labColor.b = 200 * (var_y - var_z);

    return labColor;
}

#endif
