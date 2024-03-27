if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_INSTALL_INCLUDEDIR
      "include/Scalix-${PROJECT_VERSION}"
      CACHE STRING "")
  set_property(CACHE CMAKE_INSTALL_INCLUDEDIR PROPERTY TYPE PATH)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package Scalix)

install(
  DIRECTORY include/ "${PROJECT_BINARY_DIR}/export/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT Scalix_Development)

install(
  TARGETS scalix
  EXPORT ScalixTargets
  RUNTIME #
          COMPONENT Scalix_Runtime
  LIBRARY #
          COMPONENT Scalix_Runtime NAMELINK_COMPONENT Scalix_Development
  ARCHIVE #
          COMPONENT Scalix_Development
  INCLUDES #
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

write_basic_package_version_file("${package}ConfigVersion.cmake"
                                 COMPATIBILITY SameMajorVersion)

# Allow package maintainers to freely override the path for the configs
set(SCALIX_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/${package}"
    CACHE STRING "CMake package config location relative to the install prefix")
set_property(CACHE SCALIX_INSTALL_CMAKEDIR PROPERTY TYPE PATH)
mark_as_advanced(SCALIX_INSTALL_CMAKEDIR)

install(
  FILES cmake/install-config.cmake
  DESTINATION "${SCALIX_INSTALL_CMAKEDIR}"
  RENAME "${package}Config.cmake"
  COMPONENT Scalix_Development)

install(
  FILES cmake/configure_target_for_scalix.cmake
  DESTINATION "${SCALIX_INSTALL_CMAKEDIR}"
  COMPONENT Scalix_Development)

install(
  FILES cmake/dependencies.cmake
  DESTINATION "${SCALIX_INSTALL_CMAKEDIR}"
  COMPONENT Scalix_Development)

install(
  FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
  DESTINATION "${SCALIX_INSTALL_CMAKEDIR}"
  COMPONENT Scalix_Development)

install(
  EXPORT ScalixTargets
  NAMESPACE Scalix::
  DESTINATION "${SCALIX_INSTALL_CMAKEDIR}"
  COMPONENT Scalix_Development)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
