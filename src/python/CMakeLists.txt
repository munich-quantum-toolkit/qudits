if(NOT TARGET _qudits)
  pybind11_add_module(
    _qudits
    # Prefer thin LTO if available
    THIN_LTO
    # Optimize the bindings for size
    OPT_SIZE
    # Source code goes here
    bindings.cpp)
  target_link_libraries(_qudits PRIVATE MQT::Qudits MQT::ProjectOptions MQT::ProjectWarnings)

  # Install directive for scikit-build-core
  install(
    TARGETS _qudits
    DESTINATION .
    COMPONENT mqt-qudits_Python)
endif()
