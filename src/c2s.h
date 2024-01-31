#include <math.h>

__constant__ double c2s[1*1+3*3+5*6+7*10+9*15] = {
1.0, 

1.0, 0.0, 0.0, // 0. 1. 0.
0.0, 1.0, 0.0, // 0. 0. 1.
0.0, 0.0, 1.0, // 1. 0. 0.

0.0, 1.0925484305920792, 0.0, 0.0, 0.0, 0.0, 
0.0, 0.0, 0.0, 0.0, 1.0925484305920792, 0.0, 
-0.31539156525252005, 0.0, 0.0, -0.31539156525252005, 0.0, 0.6307831305050401, 
0.0, 0.0, 1.0925484305920792, 0.0, 0.0, 0.0, 
0.5462742152960396, 0.0, 0.0, -0.5462742152960396, 0.0, 0.0, 

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


__constant__ int c2s_ptr[5] = {0, 1*1, 1*1+3*3, 1*1+3*3+5*6, 1*1+3*3+5*6+7*10};
