import pkg_resources

# 설치된 모든 패키지와 버전 출력
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

for package, version in sorted(installed_packages.items()):
    print(f"{package}: {version}")


park = 1
