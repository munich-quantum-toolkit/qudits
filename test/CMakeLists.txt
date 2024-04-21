# add unit tests
package_add_test(mqt-qudits-test mqt-qudits test_pkg.cpp)
package_add_test(mqt-qudits-example mqt-qudits main.cpp)

add_executable(mqt-qudits-collect data_collect.cpp)
target_link_libraries(mqt-qudits-collect PRIVATE mqt-qudits)
set_target_properties(mqt-qudits-collect PROPERTIES FOLDER tests)
