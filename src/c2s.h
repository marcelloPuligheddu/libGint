/*
Copyright (c) 2023 Science and Technology Facilities Council

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef C2S_H_COMPILE_GUARD
#define C2S_H_COMPILE_GUARD


#include <math.h>

static double c2s[1*1+3*3+5*6+7*10+9*15] = {
1.0, 

0.0, 1.0, 0.0, // 1.0, 0.0, 0.0, // 0. 1. 0.
0.0, 0.0, 1.0, // 0.0, 1.0, 0.0, // 0. 0. 1.
1.0, 0.0, 0.0, // 0.0, 0.0, 1.0, // 1. 0. 0.

// libcint c2s matrix for l=2
//0.0, 1.0925484305920792, 0.0, 0.0, 0.0, 0.0, 
//0.0, 0.0, 0.0, 0.0, 1.0925484305920792, 0.0, 
//-0.31539156525252005, 0.0, 0.0, -0.31539156525252005, 0.0, 0.6307831305050401, 
//0.0, 0.0, 1.0925484305920792, 0.0, 0.0, 0.0, 
//0.5462742152960396, 0.0, 0.0, -0.5462742152960396, 0.0, 0.0, 

// cp2k c2s matrix for l=2
 0., 1., 0., 0., 0., 0.,
 0., 0., 0., 0., 1., 0.,
-0.28867513,0., 0., -0.28867513, 0., 0.57735027,
 0., 0., 1., 0., 0., 0.,
 0.5, 0., 0., -0.5, 0., 0.,

// FIXME Probably wrong for cp2k !!
0.0, 1.7701307697799304, 0.0, 0.0, 0.0, 0.0, -0.5900435899266435, 0.0, 0.0, 0.0, 
0.0, 0.0, 0.0, 0.0, 2.8906114426405543, 0.0, 0.0, 0.0, 0.0, 0.0, 
0.0, -0.4570457994644657, 0.0, 0.0, 0.0, 0.0, -0.4570457994644657, 0.0, 1.8281831978578629, 0.0, 
0.0, 0.0, -1.1195289977703462, 0.0, 0.0, 0.0, 0.0, -1.1195289977703462, 0.0, 0.7463526651802308, 
-0.4570457994644657, 0.0, 0.0, -0.4570457994644657, 0.0, 1.8281831978578629, 0.0, 0.0, 0.0, 0.0, 
0.0, 0.0, 1.4453057213202771, 0.0, 0.0, 0.0, 0.0, -1.4453057213202771, 0.0, 0.0, 
0.5900435899266435, 0.0, 0.0, -1.7701307697799304, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 

0.0, 2.5033429417967046, 0.0, 0.0, 0.0, 0.0, -2.5033429417967046, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
0.0, 0.0, 0.0, 0.0, 5.310392309339791, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.7701307697799304, 0.0, 0.0, 0.0, 
0.0, -0.94617469575756, 0.0, 0.0, 0.0, 0.0, -0.94617469575756, 0.0, 5.6770481745453605, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
0.0, 0.0, 0.0, 0.0, -2.0071396306718676, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0071396306718676, 0.0, 2.676186174229157, 0.0, 
0.31735664074561293, 0.0, 0.0, 0.6347132814912259, 0.0, -2.5388531259649034, 0.0, 0.0, 0.0, 0.0, 0.31735664074561293, 0.0, -2.5388531259649034, 0.0, 0.8462843753216345, 
0.0, 0.0, -2.0071396306718676, 0.0, 0.0, 0.0, 0.0, -2.0071396306718676, 0.0, 2.676186174229157, 0.0, 0.0, 0.0, 0.0, 0.0, 
-0.47308734787878, 0.0, 0.0, 0.0, 0.0, 2.8385240872726802, 0.0, 0.0, 0.0, 0.0, 0.47308734787878, 0.0, -2.8385240872726802, 0.0, 0.0, 
0.0, 0.0, 1.7701307697799304, 0.0, 0.0, 0.0, 0.0, -5.310392309339791, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
0.6258357354491761, 0.0, 0.0, -3.755014412695057, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6258357354491761, 0.0, 0.0, 0.0, 0.0 
};


static int c2s_ptr[5] = {0, 1*1, 1*1+3*3, 1*1+3*3+5*6, 1*1+3*3+5*6+7*10};

#endif // #ifndef C2S_H_COMPILE_GUARD
