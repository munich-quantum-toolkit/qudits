DITQASM 2.0;
qreg q [3][3,4,5];
creg meas[3];
cutwo (/home/k3vn/Documents/MQTQUDIT_DEV/MQT-QUDITS-CDA/mqt-qudits/test/python/compiler/data_deb/CUt[4, 5]_DLQs.npy) q[1], q[2];
//cuone (/home/k3vn/Documents/MQTQUDIT_DEV/MQT-QUDITS-CDA/mqt-qudits/test/python/compiler/data_deb/CUo5_UEKb.npy) q[2];
//cuone (/home/k3vn/Documents/MQTQUDIT_DEV/MQT-QUDITS-CDA/mqt-qudits/test/python/compiler/data_deb/CUo5_LByK.npy) q[2];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
