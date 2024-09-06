set(MQT_QUDITS_TARGET_NAME "mqt-qudits")

if(NOT TARGET ${MQT_QUDITS_TARGET_NAME})
  # collect headers
  file(GLOB_RECURSE MQT_QUDITS_HEADERS ${MQT_QUDITS_INCLUDE_BUILD_DIR}/dd/*.hpp)

  add_library(${MQT_QUDITS_TARGET_NAME} INTERFACE ${MQT_QUDITS_HEADERS})

  # set include directories
  target_include_directories(${MQT_QUDITS_TARGET_NAME}
                             INTERFACE $<BUILD_INTERFACE:${MQT_QUDITS_INCLUDE_BUILD_DIR}>)

  # add options and warnings to the library
  target_link_libraries(${MQT_QUDITS_TARGET_NAME} INTERFACE MQT::ProjectOptions
                                                            MQT::ProjectWarnings)

  # add MQT alias
  add_library(MQT::Qudits ALIAS ${MQT_QUDITS_TARGET_NAME})
endif()

if(BUILD_MQT_QUDITS_BINDINGS)
  add_subdirectory(python)
endif()
