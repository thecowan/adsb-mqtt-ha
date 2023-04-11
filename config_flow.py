"""Tests for config flows."""
from __future__ import annotations

import logging

from homeassistant import config_entries
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import CONF_NAME, Platform
from homeassistant.core import HomeAssistant, State, callback
from homeassistant.helpers import entity_registry
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.entity_registry import EntityRegistry
from homeassistant.helpers.selector import selector
import voluptuous as vol

from .const import DEFAULT_NAME, DOMAIN
from .sensor import (
    CONF_USE_FAS_ICONS,
    CONF_ENABLED_SENSORS,
    CONF_POLL,
    CONF_SCAN_INTERVAL,
    CONF_ADSB_SENSOR,
    CONF_ADSB_JSON_ATTRIBUTE,
    POLL_DEFAULT,
    SCAN_INTERVAL_DEFAULT,
    JSON_ATTRIBUTE_DEFAULT,
    SensorType,
)

_LOGGER = logging.getLogger(__name__)


def get_value(
    config_entry: config_entries.ConfigEntry | None, param: str, default=None
):
    """Get current value for configuration parameter.

    :param config_entry: config_entries|None: config entry from Flow
    :param param: str: parameter name for getting value
    :param default: default value for parameter, defaults to None
    :returns: parameter value, or default value or None
    """
    if config_entry is not None:
        return config_entry.options.get(param, config_entry.data.get(param, default))
    else:
        return default


def build_schema(
    config_entry: config_entries | None,
    hass: HomeAssistant,
    show_advanced: bool = False,
    step: str = "user",
) -> vol.Schema:
    """Build configuration schema.

    :param config_entry: config entry for getting current parameters on None
    :param hass: Home Assistant instance
    :param show_advanced: bool: should we show advanced options?
    :param step: for which step we should build schema
    :return: Configuration schema with default parameters
    """
    entity_registry_instance = entity_registry.async_get(hass)

    schema = vol.Schema(
        {
            vol.Required(
                CONF_NAME, default=get_value(config_entry, CONF_NAME, DEFAULT_NAME)
            ): str,
            vol.Required(
                CONF_ADSB_SENSOR, default=get_value(config_entry, CONF_ADSB_SENSOR, "")
            ): selector({
                "entity": {
                    "filter": {
                        "domain": "sensor",
                    },
                },
            }),
            vol.Optional(
                CONF_ADSB_JSON_ATTRIBUTE,
                default=get_value(
                    config_entry, CONF_ADSB_JSON_ATTRIBUTE, JSON_ATTRIBUTE_DEFAULT
                ),
            ): str,
        },
    )
    if show_advanced:
        schema = schema.extend(
            {
                vol.Optional(
                    CONF_POLL, default=get_value(config_entry, CONF_POLL, POLL_DEFAULT)
                ): bool,
                vol.Optional(
                    CONF_SCAN_INTERVAL,
                    default=get_value(
                        config_entry, CONF_SCAN_INTERVAL, SCAN_INTERVAL_DEFAULT
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=1)),
                vol.Optional(
                    CONF_USE_FAS_ICONS,
                    default=get_value(config_entry, CONF_USE_FAS_ICONS, False),
                ): bool,
            }
        )
        if step == "user":
            schema = schema.extend(
                {
                    vol.Optional(
                        CONF_ENABLED_SENSORS,
                        default=list(SensorType),
                    ): cv.multi_select(
                        {
                            sensor_type: sensor_type.to_name()
                            for sensor_type in SensorType
                        }
                    ),
                }
            )

    return schema


def check_input(hass: HomeAssistant, user_input: dict) -> dict:
    """Check that we may use suggested configuration.

    :param hass: hass instance
    :param user_input: user input
    :returns: dict with error.
    """

    # ToDo: user_input have ConfigType type, but it in codebase since 2021.12.10

    result = {}

    sensor = hass.states.get(user_input[CONF_ADSB_SENSOR])

    if sensor is None:
        result["base"] = "adsb_sensor_not_found"

    return result


class AdsbInfoConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Configuration flow for setting up new adsb_info entry."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return AdsbInfoOptionsFlow(config_entry)

    async def async_step_user(self, user_input=None):
        """Handle a flow initialized by the user."""
        errors = {}

        if user_input is not None:
            if not (errors := check_input(self.hass, user_input)):
                er = entity_registry.async_get(self.hass)

                t_sensor = er.async_get(user_input[CONF_ADSB_SENSOR])
                _LOGGER.debug(f"Going to use t_sensor {t_sensor}")

                if t_sensor is not None:
                    unique_id = f"{t_sensor.unique_id}"
                    entry = await self.async_set_unique_id(unique_id)
                    if entry is not None:
                        _LOGGER.debug(f"An entry with the unique_id {unique_id} already exists: {entry.data}")
                    self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=user_input[CONF_NAME],
                    data=user_input,
                )

        schema = build_schema(
            config_entry=None,
            hass=self.hass,
            show_advanced=self.show_advanced_options,
        )

        if schema is None:
            if self.show_advanced_options:
                reason = "no_sensors_advanced"
            else:
                reason = "no_sensors"
            return self.async_abort(reason=reason)

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )


class AdsbInfoOptionsFlow(config_entries.OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage the options."""

        errors = {}
        if user_input is not None:
            _LOGGER.debug(f"OptionsFlow: going to update configuration {user_input}")
            if not (errors := check_input(self.hass, user_input)):
                return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=build_schema(
                config_entry=self.config_entry,
                hass=self.hass,
                show_advanced=self.show_advanced_options,
                step="init",
            ),
            errors=errors,
        )
