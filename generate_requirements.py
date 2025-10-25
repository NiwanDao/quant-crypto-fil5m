import pkg_resources
import subprocess
import sys
import os

def generate_requirements_file(output_file="requirements.txt"):
    try:
        # 获取当前环境中所有已安装包及其版本
        installed_packages = pkg_resources.working_set
        packages = sorted(
            ["%s==%s" % (pkg.key, pkg.version) for pkg in installed_packages]
        )

        # 写入文件
        with open(output_file, "w") as f:
            f.write("\n".join(packages))

        print(f"✅ 已生成 {output_file}，共 {len(packages)} 个包。")
    except Exception as e:
        print(f"❌ 生成 requirements.txt 失败: {e}")


if __name__ == "__main__":
    # 自动检查是否安装了 pkg_resources
    try:
        import pkg_resources
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])

    generate_requirements_file()
