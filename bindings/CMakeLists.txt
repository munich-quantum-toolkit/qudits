add_mqt_python_binding(
  QUDITS
  ${MQT_QUDITS_TARGET_NAME}-bindings
  bindings.cpp
  MODULE_NAME
  _qudits
  INSTALL_DIR
  .
  LINK_LIBS
  MQT::Qudits)

install(
  TARGETS ${MQT_QUDITS_TARGET_NAME}-bindings
  DESTINATION .
  COMPONENT ${MQT_QUDITS_TARGET_NAME}_Python)
