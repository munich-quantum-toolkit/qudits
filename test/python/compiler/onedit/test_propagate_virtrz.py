from unittest import TestCase
import numpy as np


class TestZPropagationPass(TestCase):

    """def test_tag_generator(self):
        gates = [R(np.pi, np.pi / 2, 0, 1, 3), Rz(np.pi / 3, 0, 3), R(np.pi, np.pi / 2, 0, 1, 3),
                 R(np.pi, np.pi / 2, 0, 1, 3), Rz(np.pi / 3, 0, 3)]
        tags = tag_generator(gates)

        self.assertEqual([0, 1, 1, 1, 2], tags)

    def test_propagate_z(self):
        QC = QuantumCircuit(1, 0, 3, None, False)
        QC.qreg[0] = [R(np.pi, np.pi / 3, 0, 1, 3), Rz(np.pi / 3, 0, 3), R(np.pi, np.pi / 3, 0, 1, 3),
                      R(np.pi, np.pi / 3, 0, 1, 3), Rz(np.pi / 3, 0, 3)]

        list_of_XYrots, Zseq = propagate_z(QC, 0, True)

        self.assertEqual(list_of_XYrots[1].phi, 2 * np.pi / 3)
        self.assertEqual(list_of_XYrots[2].phi, 2 * np.pi / 3)
        self.assertEqual(list_of_XYrots[0].phi, np.pi)

        self.assertEqual(Zseq[1].theta, 4 * np.pi)
        self.assertEqual(Zseq[2].theta, 4 * np.pi)
        self.assertEqual(Zseq[0].theta, 2 * np.pi / 3)

    def test_remove_z(self):
        QC = QuantumCircuit(1, 0, 3, None, False)
        QC.qreg[0] = [R(np.pi, np.pi / 3, 0, 1, 3), Rz(np.pi / 3, 0, 3), R(np.pi, np.pi / 3, 0, 1, 3),
                      R(np.pi, np.pi / 3, 0, 1, 3), Rz(np.pi / 3, 0, 3)]
        remove_Z(QC, back=True)

        self.assertIsInstance(QC.qreg[0][0], Rz)
        self.assertIsInstance(QC.qreg[0][1], Rz)
        self.assertIsInstance(QC.qreg[0][2], Rz)
        self.assertIsInstance(QC.qreg[0][3], R)
        self.assertIsInstance(QC.qreg[0][4], R)
        self.assertIsInstance(QC.qreg[0][4], R)

        QC = QuantumCircuit(1, 0, 3, None, False)
        QC.qreg[0] = [R(np.pi, np.pi / 3, 0, 1, 3), Rz(np.pi / 3, 0, 3), R(np.pi, np.pi / 3, 0, 1, 3),
                      R(np.pi, np.pi / 3, 0, 1, 3), Rz(np.pi / 3, 0, 3)]
        remove_Z(QC, False)

        self.assertIsInstance(QC.qreg[0][0], R)
        self.assertIsInstance(QC.qreg[0][1], R)
        self.assertIsInstance(QC.qreg[0][2], R)
        self.assertIsInstance(QC.qreg[0][3], Rz)
        self.assertIsInstance(QC.qreg[0][4], Rz)
        self.assertIsInstance(QC.qreg[0][4], Rz)
"""