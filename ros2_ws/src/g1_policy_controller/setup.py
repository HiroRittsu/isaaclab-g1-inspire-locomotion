from setuptools import setup

package_name = "g1_policy_controller"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ubuntu",
    maintainer_email="ubuntu@example.com",
    description="External ROS 2 policy node for G1 Isaac Sim deploy.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "policy_node = g1_policy_controller.policy_node:main",
            "action_graph_policy_node = g1_policy_controller.action_graph_policy_node:main",
        ],
    },
)
