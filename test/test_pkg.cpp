#include "dd/MDDPackage.hpp"

#include "gtest/gtest.h"

using namespace dd::literals;

TEST(DDPackageTest, RequestInvalidPackageSize) {
  EXPECT_THROW(auto dd = std::make_unique<dd::MDDPackage>(
                   std::numeric_limits<dd::QuantumRegister>::max() + 2,
                   std::vector<std::size_t>(
                       2, std::numeric_limits<dd::QuantumRegister>::max() + 2)),
               std::invalid_argument);
}

TEST(DDPackageTest, TrivialTest) {
  auto dd = std::make_unique<dd::MDDPackage>(2, std::vector<std::size_t>{3, 2});
  EXPECT_EQ(dd->qregisters(), 2);

  dd::ComplexValue const h3plus =
      dd::COMPLEX_SQRT3_3 *
      dd::ComplexValue{std::cos(2. * dd::PI / 3.), std::sin(2. * dd::PI / 3.)};
  dd::ComplexValue const h3minus =
      dd::COMPLEX_SQRT3_3 *
      dd::ComplexValue{std::cos(4. * dd::PI / 3.), std::sin(4. * dd::PI / 3.)};

  auto hGate0 = dd->makeGateDD<dd::TritMatrix>(dd::H3(), 2, 0);

  ASSERT_TRUE(dd->getValueByPath(hGate0, "00")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));
  ASSERT_TRUE(dd->getValueByPath(hGate0, "10")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));
  ASSERT_TRUE(dd->getValueByPath(hGate0, "20")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));
  ASSERT_TRUE(dd->getValueByPath(hGate0, "30")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));

  ASSERT_TRUE(dd->getValueByPath(hGate0, "40").approximatelyEquals(h3plus));

  ASSERT_TRUE(dd->getValueByPath(hGate0, "50").approximatelyEquals(h3minus));

  ASSERT_TRUE(dd->getValueByPath(hGate0, "60")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));

  ASSERT_TRUE(dd->getValueByPath(hGate0, "70").approximatelyEquals(h3minus));

  ASSERT_TRUE(dd->getValueByPath(hGate0, "80").approximatelyEquals(h3plus));

  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "01").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "11").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "21").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "31").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "41").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "51").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "61").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "71").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "81").approximatelyEquals(dd::COMPLEX_ZERO));

  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "02").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "12").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "22").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "32").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "42").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "52").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "62").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "72").approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(
      dd->getValueByPath(hGate0, "82").approximatelyEquals(dd::COMPLEX_ZERO));

  ASSERT_TRUE(dd->getValueByPath(hGate0, "03")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));
  ASSERT_TRUE(dd->getValueByPath(hGate0, "13")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));
  ASSERT_TRUE(dd->getValueByPath(hGate0, "23")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));
  ASSERT_TRUE(dd->getValueByPath(hGate0, "33")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));

  ASSERT_TRUE(dd->getValueByPath(hGate0, "43").approximatelyEquals(h3plus));
  ASSERT_TRUE(dd->getValueByPath(hGate0, "53").approximatelyEquals(h3minus));

  ASSERT_TRUE(dd->getValueByPath(hGate0, "63")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));

  ASSERT_TRUE(dd->getValueByPath(hGate0, "73").approximatelyEquals(h3minus));
  ASSERT_TRUE(dd->getValueByPath(hGate0, "83").approximatelyEquals(h3plus));

  // case with higher target
  auto hGate1 = dd->makeGateDD<dd::GateMatrix>(dd::Hmat, 2, 1);
  ASSERT_EQ(dd->getValueByPath(hGate1, "00"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "10"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "20"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "30"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "40"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "50"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "60"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "70"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "80"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));

  ASSERT_EQ(dd->getValueByPath(hGate1, "01"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "11"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "21"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "31"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "41"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "51"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "61"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "71"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "81"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));

  ASSERT_EQ(dd->getValueByPath(hGate1, "02"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "12"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "22"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "32"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "42"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "52"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "62"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "72"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "82"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));

  ASSERT_EQ(dd->getValueByPath(hGate1, "03"),
            (dd::ComplexValue{-dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "13"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "23"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "33"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "43"),
            (dd::ComplexValue{-dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "53"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "63"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "73"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(hGate1, "83"),
            (dd::ComplexValue{-dd::SQRT2_2, 0}));

  dd::Controls const controls1{{1, 1}};
  auto controlledH3 = dd->makeGateDD<dd::TritMatrix>(dd::H3(), 2, controls1, 0);

  ASSERT_TRUE(dd->getValueByPath(controlledH3, "00")
                  .approximatelyEquals(dd::COMPLEX_ONE));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "10")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "20")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "30")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "40")
                  .approximatelyEquals(dd::COMPLEX_ONE));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "50")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "60")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "70")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "80")
                  .approximatelyEquals(dd::COMPLEX_ONE));

  ASSERT_TRUE(dd->getValueByPath(controlledH3, "01")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "11")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "21")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "31")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "41")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "51")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "61")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "71")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "81")
                  .approximatelyEquals(dd::COMPLEX_ZERO));

  ASSERT_TRUE(dd->getValueByPath(controlledH3, "02")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "12")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "22")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "32")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "42")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "52")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "62")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "72")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "82")
                  .approximatelyEquals(dd::COMPLEX_ZERO));

  ASSERT_TRUE(dd->getValueByPath(controlledH3, "03")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "13")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "23")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "33")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));
  ASSERT_TRUE(
      dd->getValueByPath(controlledH3, "43").approximatelyEquals(h3plus));
  ASSERT_TRUE(
      dd->getValueByPath(controlledH3, "53").approximatelyEquals(h3minus));
  ASSERT_TRUE(dd->getValueByPath(controlledH3, "63")
                  .approximatelyEquals(dd::COMPLEX_SQRT3_3));
  ASSERT_TRUE(
      dd->getValueByPath(controlledH3, "73").approximatelyEquals(h3minus));
  ASSERT_TRUE(
      dd->getValueByPath(controlledH3, "83").approximatelyEquals(h3plus));

  dd::Controls const controls0{{0, 1}};
  auto controlled0H2 =
      dd->makeGateDD<dd::GateMatrix>(dd::Hmat, 2, controls0, 1);

  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "00")
                  .approximatelyEquals(dd::COMPLEX_ONE));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "10")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "20")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "30")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "40")
                  .approximatelyEquals(dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "50")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "60")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "70")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "80")
                  .approximatelyEquals(dd::COMPLEX_ONE));

  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "01")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "11")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "21")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "31")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "41")
                  .approximatelyEquals(dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "51")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "61")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "71")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "81")
                  .approximatelyEquals(dd::COMPLEX_ZERO));

  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "02")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "12")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "22")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "32")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "42")
                  .approximatelyEquals(dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "52")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "62")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "72")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "82")
                  .approximatelyEquals(dd::COMPLEX_ZERO));

  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "03")
                  .approximatelyEquals(dd::COMPLEX_ONE));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "13")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "23")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "33")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "43")
                  .approximatelyEquals(dd::ComplexValue{-dd::SQRT2_2, 0}));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "53")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "63")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "73")
                  .approximatelyEquals(dd::COMPLEX_ZERO));
  ASSERT_TRUE(dd->getValueByPath(controlled0H2, "83")
                  .approximatelyEquals(dd::COMPLEX_ONE));

  using namespace dd::literals;
}

TEST(DDPackageTest, Identity) {
  auto dd =
      std::make_unique<dd::MDDPackage>(4, std::vector<std::size_t>{2, 3, 2, 4});
  EXPECT_EQ(dd->qregisters(), 4);
  EXPECT_TRUE(dd->makeIdent(0).isOneTerminal());
  EXPECT_TRUE(dd->makeIdent(0, -1).isOneTerminal());

  auto id3 = dd->makeIdent(3);
  EXPECT_EQ(dd->makeIdent(0, 2), id3);
  const auto& table = dd->getIdentityTable();
  EXPECT_NE(table[2].nextNode, nullptr);

  auto id2 = dd->makeIdent(0, 1); // should be found in idTable
  EXPECT_EQ(dd->makeIdent(2), id2);

  auto id4 = dd->makeIdent(0, 3); // should use id3 and extend it
  EXPECT_EQ(dd->makeIdent(0, 3), id4);
  EXPECT_NE(table[3].nextNode, nullptr);

  auto idCached = dd->makeIdent(4);
  EXPECT_EQ(id4, idCached);
}

TEST(DDPackageTest, Multiplication) {
  auto dd =
      std::make_unique<dd::MDDPackage>(3, std::vector<std::size_t>{2, 2, 3});
  EXPECT_EQ(dd->qregisters(), 3);

  auto xGate = dd->makeGateDD<dd::GateMatrix>(dd::Xmat, 3, 0);
  auto x3dagGate = dd->makeGateDD<dd::TritMatrix>(dd::X3dag, 3, 2);
  auto x3Gate = dd->makeGateDD<dd::TritMatrix>(dd::X3, 3, 2);

  dd::Controls const control{{0, 1}, {2, 1}};
  auto ctrlxGate = dd->makeGateDD<dd::GateMatrix>(dd::Xmat, 3, control, 1);

  auto zeroState = dd->makeZeroState(3);

  auto evolution = dd->multiply(xGate, zeroState);

  evolution = dd->multiply(x3Gate, evolution);

  evolution = dd->multiply(ctrlxGate, evolution);

  evolution = dd->multiply(x3dagGate, evolution);

  auto basis110State = dd->makeBasisState(3, {1, 1, 0});

  ASSERT_EQ(dd->fidelity(zeroState, evolution), 0.0);
  ASSERT_EQ(dd->fidelity(evolution, basis110State), 1.0);
}

TEST(DDPackageTest, ConjugateTranspose) {
  const std::vector<std::size_t> lines{3};
  const dd::QuantumRegisterCount numLines = 1U;
  auto dd = std::make_unique<dd::MDDPackage>(numLines, lines);
  auto zeroState = dd->makeZeroState(numLines);
  auto h = dd->makeGateDD<dd::TritMatrix>(dd::H3(), numLines, 0);
  auto psi = dd->multiply(h, zeroState);
  psi = dd->multiply(dd->conjugateTranspose(h), psi);
  ASSERT_EQ(dd->fidelity(psi, zeroState), 1.0);
}

TEST(DDPackageTest, QutritBellState) {
  auto dd = std::make_unique<dd::MDDPackage>(2, std::vector<std::size_t>{3, 3});
  EXPECT_EQ(dd->qregisters(), 2);
  // Gates
  auto h3Gate = dd->makeGateDD<dd::TritMatrix>(dd::H3(), 2, 0);

  dd::Controls const control01{{0, 1}};
  auto ctrlx1Gate = dd->makeGateDD<dd::TritMatrix>(dd::X3, 2, control01, 1);

  dd::Controls const control02{{0, 2}};
  auto ctrlx2Gate = dd->makeGateDD<dd::TritMatrix>(dd::X3, 2, control02, 1);

  // Final Unitary
  auto op = dd->multiply(ctrlx1Gate, h3Gate);
  op = dd->multiply(ctrlx2Gate, op);
  op = dd->multiply(ctrlx2Gate, op);

  // Evolution
  auto evolution = dd->makeZeroState(2);

  evolution = dd->multiply(op, evolution);

  auto basis00State = dd->makeBasisState(2, {0, 0});
  auto basis11State = dd->makeBasisState(2, {1, 1});
  auto basis22State = dd->makeBasisState(2, {2, 2});

  ASSERT_NEAR(dd->fidelity(basis00State, evolution), 0.3333333333333333,
              dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(basis11State, evolution), 0.3333333333333333,
              dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(basis22State, evolution), 0.3333333333333333,
              dd::ComplexTable<>::tolerance());

  // Evolution gate times state

  evolution = dd->makeZeroState(2);

  evolution = dd->multiply(h3Gate, evolution);
  evolution = dd->multiply(ctrlx1Gate, evolution);
  evolution = dd->multiply(ctrlx2Gate, evolution);
  evolution = dd->multiply(ctrlx2Gate, evolution);

  ASSERT_NEAR(dd->fidelity(basis00State, evolution), 0.3333333333333333,
              dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(basis11State, evolution), 0.3333333333333333,
              dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(basis22State, evolution), 0.3333333333333333,
              dd::ComplexTable<>::tolerance());
}

TEST(DDPackageTest, W3State) {
  auto dd =
      std::make_unique<dd::MDDPackage>(3, std::vector<std::size_t>{3, 3, 3});
  EXPECT_EQ(dd->qregisters(), 3);

  auto evolution = dd->makeZeroState(3);

  auto h3Gate = dd->makeGateDD<dd::TritMatrix>(dd::H3(), 3, 1);
  dd::Controls const control10{{1, 0}};
  auto xp10 = dd->makeGateDD<dd::TritMatrix>(dd::X3, 3, control10, 0);
  dd::Controls const control12{{1, 2}};
  auto xp12 = dd->makeGateDD<dd::TritMatrix>(dd::X3, 3, control12, 2);

  auto csum21 = dd->csum(3, 2, 1, true);

  evolution = dd->multiply(h3Gate, evolution);
  evolution = dd->multiply(xp10, evolution);
  evolution = dd->multiply(xp12, evolution);
  evolution = dd->multiply(csum21, evolution);
  evolution = dd->multiply(csum21, evolution);
}
TEST(DDPackageTest, Mix23WState) {
  auto dd = std::make_unique<dd::MDDPackage>(
      6, std::vector<std::size_t>{2, 3, 3, 2, 3, 3});
  EXPECT_EQ(dd->qregisters(), 6);

  auto evolution = dd->makeBasisState(6, {1, 0, 0, 0, 0, 0});

  evolution = dd->spread2(6, std::vector<dd::QuantumRegister>{0, 3}, evolution);
  evolution =
      dd->spread3(6, std::vector<dd::QuantumRegister>{0, 1, 2}, evolution);
  evolution =
      dd->spread3(6, std::vector<dd::QuantumRegister>{3, 4, 5}, evolution);

  ASSERT_NEAR(
      dd->fidelity(dd->makeBasisState(6, {1, 0, 0, 0, 0, 0}), evolution),
      1.0 / 6.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(
      dd->fidelity(dd->makeBasisState(6, {0, 1, 0, 0, 0, 0}), evolution),
      1.0 / 6.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(
      dd->fidelity(dd->makeBasisState(6, {0, 0, 1, 0, 0, 0}), evolution),
      1.0 / 6.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(
      dd->fidelity(dd->makeBasisState(6, {0, 0, 0, 1, 0, 0}), evolution),
      1.0 / 6.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(
      dd->fidelity(dd->makeBasisState(6, {0, 0, 0, 0, 1, 0}), evolution),
      1.0 / 6.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(
      dd->fidelity(dd->makeBasisState(6, {0, 0, 0, 0, 0, 1}), evolution),
      1.0 / 6.0, dd::ComplexTable<>::tolerance());
}

TEST(DDPackageTest, W35State) {
  auto dd = std::make_unique<dd::MDDPackage>(
      15,
      std::vector<std::size_t>{3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5});
  EXPECT_EQ(dd->qregisters(), 15);

  auto evolution =
      dd->makeBasisState(15, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  evolution =
      dd->spread3(15, std::vector<dd::QuantumRegister>{0, 1, 2}, evolution);
  evolution = dd->spread5(15, std::vector<dd::QuantumRegister>{0, 3, 4, 5, 6},
                          evolution);
  evolution = dd->spread5(15, std::vector<dd::QuantumRegister>{1, 7, 8, 9, 10},
                          evolution);
  evolution = dd->spread5(
      15, std::vector<dd::QuantumRegister>{2, 11, 12, 13, 14}, evolution);

  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                                   0, 0, 0, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                                   0, 0, 0, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                                   0, 0, 0, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                   0, 0, 0, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   1, 0, 0, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 1, 0, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 1, 0, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 1, 0}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(dd->makeBasisState(15, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                   0, 0, 0, 0, 1}),
                           evolution),
              1.0 / 15.0, dd::ComplexTable<>::tolerance());
}

TEST(DDPackageTest, W5State) {
  auto dd = std::make_unique<dd::MDDPackage>(
      5, std::vector<std::size_t>{5, 5, 5, 5, 5});

  EXPECT_EQ(dd->qregisters(), 5);

  auto evolution = dd->makeZeroState(5);

  auto h5Gate = dd->makeGateDD<dd::QuintMatrix>(dd::H5(), 5, 1);

  dd::Controls const control10{{1, 0}};
  auto xp10 = dd->makeGateDD<dd::QuintMatrix>(dd::X5, 5, control10, 0);
  dd::Controls const control12{{1, 2}};
  auto xp12 = dd->makeGateDD<dd::QuintMatrix>(dd::X5, 5, control12, 2);
  dd::Controls const control13{{1, 3}};
  auto xp13 = dd->makeGateDD<dd::QuintMatrix>(dd::X5, 5, control13, 3);
  dd::Controls const control14{{1, 4}};
  auto xp14 = dd->makeGateDD<dd::QuintMatrix>(dd::X5, 5, control14, 4);

  auto csum21 = dd->csum(5, 2, 1, true);
  auto csum31 = dd->csum(5, 3, 1, true);
  auto csum41 = dd->csum(5, 4, 1, true);

  evolution = dd->multiply(h5Gate, evolution);
  evolution = dd->multiply(xp10, evolution);
  evolution = dd->multiply(xp12, evolution);
  evolution = dd->multiply(csum21, evolution);
  evolution = dd->multiply(csum21, evolution);
  evolution = dd->multiply(xp13, evolution);
  evolution = dd->multiply(csum31, evolution);
  evolution = dd->multiply(csum31, evolution);
  evolution = dd->multiply(csum31, evolution);
  evolution = dd->multiply(xp14, evolution);
  evolution = dd->multiply(csum41, evolution);
  evolution = dd->multiply(csum41, evolution);
  evolution = dd->multiply(csum41, evolution);
  evolution = dd->multiply(csum41, evolution);
}

TEST(DDPackageTest, FullMixWState) {
  std::vector<size_t> orderOfLayers{2, 3, 2, 3, 3};
  std::vector<std::size_t> lines{};
  dd::QuantumRegisterCount numLines = 0U;
  std::map<std::size_t, std::vector<std::size_t>> application;
  std::vector<std::vector<dd::QuantumRegister>> indexes{{}};
  std::size_t sizeTracker = 0;

  bool initial = true;
  for (auto i = 0U; i < orderOfLayers.size(); i++) {
    // Update cardinality of each line, put them in order
    // Update the application indexes
    if (initial) {
      for (auto j = 0U; j < orderOfLayers.at(i); j++) {
        lines.push_back(orderOfLayers.at(i));
        indexes.at(0).push_back(static_cast<dd::QuantumRegister>(j));
        numLines++;
      }
      application[i] = std::vector<std::size_t>{sizeTracker};
      application[i + 1] = {};
      sizeTracker++;
      for (auto j = 0U; j < indexes.at(0).size(); j++) {
        indexes.push_back(std::vector<dd::QuantumRegister>{
            indexes.at(0).at(static_cast<std::size_t>(j))});
        application[i + 1].push_back(sizeTracker);
        sizeTracker++;
      }

      initial = false;
    } else {
      auto tempLine = lines.size();
      auto counter = 0U;
      for (auto k = 0U; k < tempLine; k++) {
        auto adder = k + counter;
        for (auto j = 1U; j < orderOfLayers.at(i); j++) {
          lines.insert(
              lines.begin() + adder + j,
              orderOfLayers.at(i)); // check if eventually gets out of range
          numLines++;
          counter++;
        }
      }
      for (auto& index : indexes) {
        for (auto& f : index) {
          f = static_cast<dd::QuantumRegister>(
              f +
              f * (static_cast<dd::QuantumRegister>(orderOfLayers.at(i) - 1)));
        }
      }
      std::vector<dd::QuantumRegister> toAdd{};
      for (auto& index : indexes) {
        if (index.size() == 1) {
          toAdd.push_back(index.at(0));
          for (auto j = 1U; j < orderOfLayers.at(i); j++) {
            index.push_back(static_cast<dd::QuantumRegister>(
                index.at(0) + static_cast<dd::QuantumRegister>(j)));
            toAdd.push_back(static_cast<dd::QuantumRegister>(
                index.at(0) + static_cast<dd::QuantumRegister>(j)));
          }
        }
      }
      if (i < orderOfLayers.size() - 1) {
        for (const dd::QuantumRegister& l : toAdd) {
          indexes.push_back(std::vector<dd::QuantumRegister>{l});
          application[i + 1].push_back(sizeTracker);
          sizeTracker++;
        }
        application[i + 2] = {};
      }
    }
  }

  auto dd = std::make_unique<dd::MDDPackage>(numLines, lines);
  EXPECT_EQ(dd->qregisters(), numLines);

  std::vector<size_t> initState(numLines, 0);
  initState.at(0) = 1;
  auto evolution = dd->makeBasisState(numLines, initState);

  for (auto i = 0U; i < orderOfLayers.size(); i++) {
    if (orderOfLayers.at(i) == 2) {
      for (auto g = 0U; g < application[i].size(); g++) {
        std::vector<dd::QuantumRegister> const inputLines =
            indexes.at(application[i].at(g));
        evolution = dd->spread2(numLines, inputLines, evolution);
      }
    } else if (orderOfLayers.at(i) == 3) {
      for (auto g = 0U; g < application[i].size(); g++) {
        evolution = dd->spread3(
            numLines,
            reinterpret_cast<const std::vector<dd::QuantumRegister>&>(
                indexes.at(application[i].at(g))),
            evolution);
      }
    } else if (orderOfLayers.at(i) == 5) {
      for (auto g = 0U; g < application[i].size(); g++) {
        evolution = dd->spread5(
            numLines,
            reinterpret_cast<const std::vector<dd::QuantumRegister>&>(
                indexes.at(application[i].at(g))),
            evolution);
      }
    }
  }

  for (auto h = 0U; h < numLines; h++) {
    std::vector<size_t> checkState(numLines, 0);
    checkState.at(h) = 1;
    ASSERT_NEAR(
        dd->fidelity(dd->makeBasisState(numLines, checkState), evolution),
        1.0 / numLines, dd::ComplexTable<>::tolerance());
  }
}

TEST(DDPackageTest, GHZQutritState) {
  auto dd =
      std::make_unique<dd::MDDPackage>(3, std::vector<std::size_t>{3, 3, 3});
  EXPECT_EQ(dd->qregisters(), 3);
  // Gates
  auto h3Gate = dd->makeGateDD<dd::TritMatrix>(dd::H3(), 3, 0);

  dd::Controls const control01{{0, 1}};
  auto cX011 = dd->makeGateDD<dd::TritMatrix>(dd::X3, 3, control01, 1);
  dd::Controls const control02{{0, 2}};
  auto cX021 = dd->makeGateDD<dd::TritMatrix>(dd::X3dag, 3, control02, 1);

  dd::Controls const control011{{0, 1}, {1, 1}};
  auto cXc01l1t2 = dd->makeGateDD<dd::TritMatrix>(dd::X3, 3, control011, 2);
  dd::Controls const control012{{0, 2}, {1, 2}};
  auto cX0122 = dd->makeGateDD<dd::TritMatrix>(dd::X3dag, 3, control012, 2);

  // auto testvec = dd->getVectorizedMatrix(cX0122);

  // Evolution
  auto evolution = dd->makeZeroState(3);

  evolution = dd->multiply(h3Gate, evolution);

  evolution = dd->multiply(cX011, evolution);
  evolution = dd->multiply(cX021, evolution);

  evolution = dd->multiply(cXc01l1t2, evolution);

  evolution = dd->multiply(cX0122, evolution);

  auto basis00State = dd->makeBasisState(3, {0, 0, 0});
  auto basis11State = dd->makeBasisState(3, {1, 1, 1});
  auto basis22State = dd->makeBasisState(3, {2, 2, 2});

  ASSERT_NEAR(dd->fidelity(basis00State, evolution), 0.3333333333333333,
              dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(basis11State, evolution), 0.3333333333333333,
              dd::ComplexTable<>::tolerance());
  ASSERT_NEAR(dd->fidelity(basis22State, evolution), 0.3333333333333333,
              dd::ComplexTable<>::tolerance());
}

TEST(DDPackageTest, GHZQutritStateScaled) {
  for (dd::QuantumRegisterCount i = 120U; i < 129U; i++) {
    const std::vector<std::size_t> init(i, 3);
    auto dd = std::make_unique<dd::MDDPackage>(i, init);
    EXPECT_EQ(dd->qregisters(), i);

    // Gates
    auto h3Gate = dd->makeGateDD<dd::TritMatrix>(dd::H3(), i, 0);
    std::vector<dd::MDDPackage::mEdge> gates = {};

    for (int target = 1; target < i; target++) {
      dd::Controls target1{};
      dd::Controls target2{};

      for (int control = 0; control < target; control++) {
        const dd::Control c1{static_cast<dd::QuantumRegister>(control), 1};
        const dd::Control c2{static_cast<dd::QuantumRegister>(control), 2};
        target1.insert(c1);
        target2.insert(c2);
      }

      gates.push_back(dd->makeGateDD<dd::TritMatrix>(
          dd::X3, i, target1, static_cast<dd::QuantumRegister>(target)));
      gates.push_back(dd->makeGateDD<dd::TritMatrix>(
          dd::X3dag, i, target2, static_cast<dd::QuantumRegister>(target)));
    }

    auto evolution = dd->makeZeroState(i);
    evolution = dd->multiply(h3Gate, evolution);

    for (auto& gate : gates) {
      evolution = dd->multiply(gate, evolution);
    }
    if (dd->qregisters() < 10) {
      std::cout << "\n" << std::endl;
      dd->printVector(evolution);
    }

    auto basis00State = dd->makeBasisState(i, std::vector<size_t>(i, 0));
    auto basis11State = dd->makeBasisState(i, std::vector<size_t>(i, 1));
    auto basis22State = dd->makeBasisState(i, std::vector<size_t>(i, 2));

    ASSERT_NEAR(dd->fidelity(basis00State, evolution), 0.3333333333333333,
                dd::ComplexTable<>::tolerance());
    ASSERT_NEAR(dd->fidelity(basis11State, evolution), 0.3333333333333333,
                dd::ComplexTable<>::tolerance());
    ASSERT_NEAR(dd->fidelity(basis22State, evolution), 0.3333333333333333,
                dd::ComplexTable<>::tolerance());
  }
}

TEST(DDPackageTest, RandomCircuits) {
  const dd::QuantumRegisterCount width = 3;
  const std::size_t depth = 1000;
  const std::size_t maxD = 5;

  std::mt19937 gen(12345); // NOLINT(cert-msc51-cpp): seed the generator with
                           // fixed value for reproducibility

  std::vector<std::size_t> particles = {};

  std::uniform_int_distribution<std::size_t> dimdistr(2, maxD);
  particles.reserve(width);
  for (auto i = 0; i < width; i++) {
    particles.emplace_back(dimdistr(gen));
  }

  auto dd = std::make_unique<dd::MDDPackage>(width, particles);

  EXPECT_EQ(dd->qregisters(), width);

  auto evolution = dd->makeZeroState(width);

  std::cout << "\n"
            << "STARTED" << "\n"
            << std::endl;

  std::uniform_int_distribution<> pickbool(0, 1);
  std::uniform_int_distribution<unsigned int> pickcontrols(1, width - 1);
  std::uniform_real_distribution<> angles(0.0, 2. * dd::PI);

  for (std::size_t timeStep = 0; timeStep < depth; timeStep++) {
    for (std::size_t line = 0; line < width; line++) {
      // chose if local gate or entangling gate
      auto randomChoice = pickbool(gen);

      if (randomChoice == 0) { // local op

        auto localChoice = pickbool(gen);

        if (localChoice == 0) { // hadamard
          std::cout << "\n"
                    << "hadamard" << "\n"
                    << std::endl;
          if (particles.at(line) == 2) {
            auto chosenGate = dd->makeGateDD<dd::GateMatrix>(
                dd::H(), width, static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          } else if (particles.at(line) == 3) {
            auto chosenGate = dd->makeGateDD<dd::TritMatrix>(
                dd::H3(), width, static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          } else if (particles.at(line) == 4) {
            auto chosenGate = dd->makeGateDD<dd::QuartMatrix>(
                dd::H4(), width, static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          } else if (particles.at(line) == 5) {
            auto chosenGate = dd->makeGateDD<dd::QuintMatrix>(
                dd::H5(), width, static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          }
        } else { // givens
          std::cout << "\n"
                    << "givens" << "\n"
                    << std::endl;
          if (particles.at(line) == 2) {
            double const theta = 0.;
            double const phi = 0.;
            auto chosenGate = dd->makeGateDD<dd::GateMatrix>(
                dd::RXY(theta, phi), width,
                static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          } else if (particles.at(line) == 3) {
            double const theta = angles(gen);
            double const phi = angles(gen);
            std::uniform_int_distribution<std::size_t> picklevel(0, 2);

            auto levelA = picklevel(gen);
            auto levelB = (levelA + 1) % 3;
            if (levelA > levelB) {
              auto temp = levelA;
              levelA = levelB;
              levelB = temp;
            }

            auto chosenGate = dd->makeGateDD<dd::TritMatrix>(
                dd::RXY3(theta, phi, levelA, levelB), width,
                static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          } else if (particles.at(line) == 4) {
            double const theta = angles(gen);
            double const phi = angles(gen);
            std::uniform_int_distribution<std::size_t> picklevel(0, 3);

            auto levelA = picklevel(gen);
            auto levelB = (levelA + 1) % 4;
            if (levelA > levelB) {
              auto temp = levelA;
              levelA = levelB;
              levelB = temp;
            }

            auto chosenGate = dd->makeGateDD<dd::QuartMatrix>(
                dd::RXY4(theta, phi, levelA, levelB), width,
                static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          } else if (particles.at(line) == 5) {
            double const theta = angles(gen);
            double const phi = angles(gen);
            std::uniform_int_distribution<std::size_t> picklevel(0, 4);

            auto levelA = picklevel(gen);
            auto levelB = (levelA + 1) % 5;
            if (levelA > levelB) {
              auto temp = levelA;
              levelA = levelB;
              levelB = temp;
            }

            auto chosenGate = dd->makeGateDD<dd::QuintMatrix>(
                dd::RXY5(theta, phi, levelA, levelB), width,
                static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          }
        }
      } else { // entangling
        std::cout << "\n"
                  << "entangling" << "\n"
                  << std::endl;

        auto entChoice = pickbool(gen);
        auto numberOfControls = pickcontrols(gen);

        std::vector<std::size_t> controlLines;

        for (auto i = 0U; i < width; i++) {
          if (i != line) {
            controlLines.push_back(i);
          }
        }

        std::shuffle(begin(controlLines), end(controlLines), gen);
        std::vector<std::size_t> controlParticles(
            controlLines.begin(), controlLines.begin() + numberOfControls);
        std::sort(controlParticles.begin(), controlParticles.end());

        dd::Controls control{};
        for (std::size_t i = 0; i < numberOfControls; i++) {
          std::uniform_int_distribution<std::size_t> picklevel(
              0, particles.at(controlParticles.at(i)) - 1);
          auto level = picklevel(gen);

          const dd::Control c{
              static_cast<dd::QuantumRegister>(controlParticles.at(i)),
              static_cast<dd::Control::Type>(level)};
          control.insert(c);
        }

        if (entChoice == 0) { // CEX based
          // selection of controls
          std::cout << "\n"
                    << "CEX" << "\n"
                    << std::endl;
          if (particles.at(line) == 2) {
            double const theta = angles(gen);
            double const phi = angles(gen);
            auto chosenGate = dd->makeGateDD<dd::GateMatrix>(
                dd::RXY(theta, phi), width, control,
                static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          } else if (particles.at(line) == 3) {
            std::uniform_int_distribution<std::size_t> picklevel(0, 2);

            double const theta = angles(gen);
            double const phi = angles(gen);
            auto levelA = picklevel(gen);
            auto levelB = (levelA + 1) % 3;
            if (levelA > levelB) {
              auto temp = levelA;
              levelA = levelB;
              levelB = temp;
            }

            auto chosenGate = dd->makeGateDD<dd::TritMatrix>(
                dd::RXY3(theta, phi, levelA, levelB), width, control,
                static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          } else if (particles.at(line) == 4) {
            std::uniform_int_distribution<std::size_t> picklevel(0, 3);

            double const theta = angles(gen);
            double const phi = angles(gen);
            auto levelA = picklevel(gen);
            auto levelB = (levelA + 1) % 4;
            if (levelA > levelB) {
              auto temp = levelA;
              levelA = levelB;
              levelB = temp;
            }

            auto chosenGate = dd->makeGateDD<dd::QuartMatrix>(
                dd::RXY4(theta, phi, levelA, levelB), width, control,
                static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          } else if (particles.at(line) == 5) {
            std::uniform_int_distribution<std::size_t> picklevel(0, 4);

            double const theta = angles(gen);
            double const phi = angles(gen);
            auto levelA = picklevel(gen);
            auto levelB = (levelA + 1) % 5;
            if (levelA > levelB) {
              auto temp = levelA;
              levelA = levelB;
              levelB = temp;
            }

            auto chosenGate = dd->makeGateDD<dd::QuintMatrix>(
                dd::RXY5(theta, phi, levelA, levelB), width, control,
                static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          }
        } else { // Controlled clifford
          std::cout << "\n"
                    << "clifford" << "\n"
                    << std::endl;
          if (particles.at(line) == 2) {
            auto chosenGate = dd->makeGateDD<dd::GateMatrix>(
                dd::Xmat, width, control,
                static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          } else if (particles.at(line) == 3) {
            auto chosenGate = dd->makeGateDD<dd::TritMatrix>(
                dd::X3, width, control, static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          } else if (particles.at(line) == 4) {
            auto chosenGate = dd->makeGateDD<dd::QuartMatrix>(
                dd::X4, width, control, static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          } else if (particles.at(line) == 5) {
            auto chosenGate = dd->makeGateDD<dd::QuintMatrix>(
                dd::X5, width, control, static_cast<dd::QuantumRegister>(line));
            evolution = dd->multiply(chosenGate, evolution);
          }
        }
      }
    }
  }
}
