"""
Tests for rosie.tool_registry — schema completeness and dispatch integrity.
"""

from __future__ import annotations

from rosie.tool_registry import TOOL_DEFINITIONS, TOOL_FUNCTIONS


class TestToolDefinitions:
    def test_all_definitions_have_required_fields(self):
        for defn in TOOL_DEFINITIONS:
            assert defn["type"] == "function"
            fn = defn["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"

    def test_all_definitions_have_dispatch_entry(self):
        """Every schema must map to a callable in TOOL_FUNCTIONS."""
        schema_names = {d["function"]["name"] for d in TOOL_DEFINITIONS}
        dispatch_names = set(TOOL_FUNCTIONS.keys())
        missing = schema_names - dispatch_names
        assert not missing, f"Schemas without dispatch entries: {missing}"

    def test_all_dispatch_entries_have_definitions(self):
        """Every dispatch entry must have a matching schema."""
        schema_names = {d["function"]["name"] for d in TOOL_DEFINITIONS}
        dispatch_names = set(TOOL_FUNCTIONS.keys())
        extra = dispatch_names - schema_names
        assert not extra, f"Dispatch entries without schemas: {extra}"

    def test_expected_tool_count(self):
        assert len(TOOL_DEFINITIONS) == 28
        assert len(TOOL_FUNCTIONS) == 28

    def test_no_duplicate_names(self):
        names = [d["function"]["name"] for d in TOOL_DEFINITIONS]
        assert len(names) == len(set(names)), "Duplicate tool names found"


class TestToolFunctionsCallable:
    def test_all_are_callable(self):
        for name, fn in TOOL_FUNCTIONS.items():
            assert callable(fn), f"{name} is not callable"

    def test_discovery_tools_exist(self):
        expected = {
            "ros2_list_topics",
            "ros2_topic_info",
            "ros2_list_services",
            "ros2_list_actions",
            "ros2_list_nodes",
            "ros2_node_info",
            "ros2_describe_interface",
        }
        assert expected.issubset(TOOL_FUNCTIONS.keys())

    def test_observation_tools_exist(self):
        expected = {"ros2_read_topic", "ros2_read_topic_stream"}
        assert expected.issubset(TOOL_FUNCTIONS.keys())

    def test_command_tools_exist(self):
        expected = {
            "ros2_publish",
            "ros2_publish_repeated",
            "ros2_call_service",
            "ros2_send_action_goal",
        }
        assert expected.issubset(TOOL_FUNCTIONS.keys())

    def test_param_tools_exist(self):
        expected = {"ros2_get_param", "ros2_set_param"}
        assert expected.issubset(TOOL_FUNCTIONS.keys())

    def test_heart_tools_exist(self):
        expected = {
            "heart_get_state",
            "heart_get_status",
            "heart_set_velocity",
            "heart_go_to_position",
            "heart_stop",
            "heart_switch_controller",
            "heart_set_trajectory",
        }
        assert expected.issubset(TOOL_FUNCTIONS.keys())

    def test_heart_config_tools_exist(self):
        expected = {"heart_auto_configure", "heart_configure"}
        assert expected.issubset(TOOL_FUNCTIONS.keys())

    def test_px4_tools_exist(self):
        expected = {"px4_arm", "px4_disarm", "px4_offboard_mode", "px4_engage"}
        assert expected.issubset(TOOL_FUNCTIONS.keys())


class TestSchemaParameterConsistency:
    """Verify that required parameters in schemas match function signatures."""

    def test_topic_info_requires_topic_name(self):
        defn = _find_defn("ros2_topic_info")
        assert "topic_name" in defn["function"]["parameters"]["required"]

    def test_publish_requires_all_three(self):
        defn = _find_defn("ros2_publish")
        required = set(defn["function"]["parameters"]["required"])
        assert required == {"topic_name", "msg_type", "data"}

    def test_call_service_requires_all_three(self):
        defn = _find_defn("ros2_call_service")
        required = set(defn["function"]["parameters"]["required"])
        assert required == {"service_name", "service_type", "request"}

    def test_send_action_goal_requires_all_three(self):
        defn = _find_defn("ros2_send_action_goal")
        required = set(defn["function"]["parameters"]["required"])
        assert required == {"action_name", "action_type", "goal"}

    def test_parameterless_tools_have_empty_required(self):
        for name in ["ros2_list_topics", "ros2_list_services", "ros2_list_actions", "ros2_list_nodes"]:
            defn = _find_defn(name)
            assert defn["function"]["parameters"]["required"] == []


def _find_defn(name: str) -> dict:
    for d in TOOL_DEFINITIONS:
        if d["function"]["name"] == name:
            return d
    raise KeyError(f"No definition found for {name}")
