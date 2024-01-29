"""Sensor platform for adsb_info."""
from asyncio import Lock
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
import json
import logging
import math
from typing import Any
from typing import Optional
from typing import Tuple

import great_circle_calculator.great_circle_calculator as gcc

from homeassistant import util
from homeassistant.backports.enum import StrEnum
from homeassistant.components.sensor import (
    DOMAIN as SENSOR_DOMAIN,
    PLATFORM_SCHEMA,
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_FRIENDLY_NAME,
    CONF_ICON_TEMPLATE,
    CONF_NAME,
    CONF_SENSORS,
    CONF_UNIQUE_ID,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfLength,
    UnitOfSpeed,
    DEGREE,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.components import mqtt
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import entity_registry
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.template import Template
from homeassistant.loader import async_get_custom_components
import voluptuous as vol

from .const import DEFAULT_NAME, DOMAIN

_LOGGER = logging.getLogger(__name__)

CONF_ENABLED_SENSORS = "enabled_sensors"
CONF_SENSOR_TYPES = "sensor_types"
CONF_USE_FAS_ICONS = "use_fas_icons"
CONF_SCAN_INTERVAL = "scan_interval"
CONF_DATA_SOURCE = "data_source"
CONF_MQTT_TOPIC = "mqtt_topic"

CONF_ADSB_SENSOR = "adsb_sensor"
CONF_ADSB_JSON_ATTRIBUTE = "adsb_json_attribute"
CONF_POLL = "poll"
# Default values
POLL_DEFAULT = False
SCAN_INTERVAL_DEFAULT = 30
JSON_ATTRIBUTE_DEFAULT = "data"
DATA_SOURCE_DEFAULT = "SENSOR"


class DataSourceType(StrEnum):
    MQTT = "MQTT"
    SENSOR = "SENSOR"

    @classmethod
    def from_string(cls, string: str) -> "DataSourceType":
        if string in list(cls):
            return cls(string)
        else:
            raise ValueError(
                f"Unknown data source type: {string}. Please check https://github.com/thecowan/adsb_info/blob/master/documentation/yaml.md for valid options."
            )


class SensorType(StrEnum):
    """Sensor type enum."""

    TRACKED_COUNT = "tracked_count"
    CLOSEST_AIRCRAFT = "closest_aircraft"
    CLOSEST_AIRCRAFT_GROUND_SPEED = "closest_aircraft_ground_speed"
    CLOSEST_AIRCRAFT_HEADING = "closest_aircraft_heading"
    CLOSEST_AIRCRAFT_BAROMETRIC_ALTITUDE = "closest_aircraft_barometric_altitude"
    CLOSEST_AIRCRAFT_DISTANCE = "closest_aircraft_distance"
    CLOSEST_AIRCRAFT_BEARING = "closest_aircraft_bearing"
    CLOSEST_AIRCRAFT_APPROACHING = "closest_aircraft_approaching"
    CLOSEST_AIRCRAFT_CPA = "closest_aircraft_cpa"

    def to_name(self) -> str:
        """Return the title of the sensor type."""
        return self.value.replace("_", " ").capitalize()

    def default_icon(self) -> str:
        """Return the default icon to use."""
        return ENTITY_ICONS[self]

    @classmethod
    def from_string(cls, string: str) -> "SensorType":
        """Return the sensor type from string."""
        if string in list(cls):
            return cls(string)
        else:
            raise ValueError(
                f"Unknown sensor type: {string}. Please check https://github.com/thecowan/adsb_info/blob/master/documentation/yaml.md for valid options."
            )

class ApproachingState(StrEnum):
    APPROACHING = "approaching"
    RECEDING = "receding"

ENTITY_ICONS = {
    SensorType.TRACKED_COUNT: "mdi:airplane",
    SensorType.CLOSEST_AIRCRAFT: "mdi:airplane-marker",
    SensorType.CLOSEST_AIRCRAFT_GROUND_SPEED: "mdi:speedometer",
    SensorType.CLOSEST_AIRCRAFT_HEADING: "mdi:compass",
    SensorType.CLOSEST_AIRCRAFT_BAROMETRIC_ALTITUDE: "mdi:ruler",
    SensorType.CLOSEST_AIRCRAFT_DISTANCE: "mdi:ruler",
    SensorType.CLOSEST_AIRCRAFT_BEARING: "mdi:compass-rose",
    SensorType.CLOSEST_AIRCRAFT_APPROACHING: "mdi:sign-direction",
    SensorType.CLOSEST_AIRCRAFT_CPA: "mdi:map-marker-radius",
}

FAS_ENTITY_ICONS = {
    SensorType.CLOSEST_AIRCRAFT_BAROMETRIC_ALTITUDE: "fas:ruler-vertical",
    SensorType.CLOSEST_AIRCRAFT_DISTANCE: "fas:ruler-horizontal",
    SensorType.CLOSEST_AIRCRAFT_APPROACHING: "fas:map-signs",
}

SENSOR_TYPES = {
    SensorType.TRACKED_COUNT: {
        "key": SensorType.TRACKED_COUNT,
        "name": SensorType.TRACKED_COUNT.to_name(),
        "icon": SensorType.TRACKED_COUNT.default_icon(),
        "state_class": SensorStateClass.MEASUREMENT,
    },
    SensorType.CLOSEST_AIRCRAFT: {
        "key": SensorType.CLOSEST_AIRCRAFT,
        "name": SensorType.CLOSEST_AIRCRAFT.to_name(),
        "icon": SensorType.CLOSEST_AIRCRAFT.default_icon(),
        "state_class": None,
        "translation_key": SensorType.CLOSEST_AIRCRAFT,
    },
    SensorType.CLOSEST_AIRCRAFT_GROUND_SPEED: {
        "key": SensorType.CLOSEST_AIRCRAFT_GROUND_SPEED,
        "name": SensorType.CLOSEST_AIRCRAFT_GROUND_SPEED.to_name(),
        "icon": SensorType.CLOSEST_AIRCRAFT_GROUND_SPEED.default_icon(),
        "state_class": SensorStateClass.MEASUREMENT,
        "device_class": SensorDeviceClass.SPEED,
        "native_unit_of_measurement": UnitOfSpeed.KNOTS,
        "suggested_display_precision": 1,
    },
    SensorType.CLOSEST_AIRCRAFT_HEADING: {
        "key": SensorType.CLOSEST_AIRCRAFT_HEADING,
        "name": SensorType.CLOSEST_AIRCRAFT_HEADING.to_name(),
        "icon": SensorType.CLOSEST_AIRCRAFT_HEADING.default_icon(),
        "state_class": SensorStateClass.MEASUREMENT,
        "native_unit_of_measurement": DEGREE,
        "suggested_display_precision": 1,
    },
    SensorType.CLOSEST_AIRCRAFT_BAROMETRIC_ALTITUDE: {
        "key": SensorType.CLOSEST_AIRCRAFT_BAROMETRIC_ALTITUDE,
        "name": SensorType.CLOSEST_AIRCRAFT_BAROMETRIC_ALTITUDE.to_name(),
        "icon": SensorType.CLOSEST_AIRCRAFT_BAROMETRIC_ALTITUDE.default_icon(),
        "state_class": SensorStateClass.MEASUREMENT,
        "device_class": SensorDeviceClass.DISTANCE,
        "native_unit_of_measurement": UnitOfLength.FEET,
        "suggested_display_precision": 0,
    },
    SensorType.CLOSEST_AIRCRAFT_DISTANCE: {
        "key": SensorType.CLOSEST_AIRCRAFT_DISTANCE,
        "name": SensorType.CLOSEST_AIRCRAFT_DISTANCE.to_name(),
        "icon": SensorType.CLOSEST_AIRCRAFT_DISTANCE.default_icon(),
        "state_class": SensorStateClass.MEASUREMENT,
        "device_class": SensorDeviceClass.DISTANCE,
        "native_unit_of_measurement": UnitOfLength.KILOMETERS,
        "suggested_display_precision": 3,
    },
    SensorType.CLOSEST_AIRCRAFT_BEARING: {
        "key": SensorType.CLOSEST_AIRCRAFT_BEARING,
        "name": SensorType.CLOSEST_AIRCRAFT_BEARING.to_name(),
        "icon": SensorType.CLOSEST_AIRCRAFT_BEARING.default_icon(),
        "state_class": SensorStateClass.MEASUREMENT,
        "native_unit_of_measurement": DEGREE,
        "suggested_display_precision": 1,
    },
    SensorType.CLOSEST_AIRCRAFT_APPROACHING: {
        "key": SensorType.CLOSEST_AIRCRAFT_APPROACHING,
        "name": SensorType.CLOSEST_AIRCRAFT_APPROACHING.to_name(),
        "icon": SensorType.CLOSEST_AIRCRAFT_APPROACHING.default_icon(),
        "device_class": SensorDeviceClass.ENUM,
        "translation_key": SensorType.CLOSEST_AIRCRAFT_APPROACHING,
        "options": list(map(str, ApproachingState)),
    },
    SensorType.CLOSEST_AIRCRAFT_CPA: {
        "key": SensorType.CLOSEST_AIRCRAFT_CPA,
        "name": SensorType.CLOSEST_AIRCRAFT_CPA.to_name(),
        "icon": SensorType.CLOSEST_AIRCRAFT_CPA.default_icon(),
        "state_class": SensorStateClass.MEASUREMENT,
        "native_unit_of_measurement": UnitOfLength.KILOMETERS,
        "suggested_display_precision": 3,
    },
}

# TODO - translate
BEARINGS = ['N', 'NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
BEARINGS_FINE = ['N','NbE','NNE','NEbN','NE','NEbE','ENE','EbN','E','EbS','ESE','SEbE','SE','SEbS','SSE','SbE','S','SbW','SSW','SWbS','SW','SWbW','WSW','WbS','W','WbN','WNW','NWbW','NW','NWbN','NNW','NbW']
BEARINGS_CRUDE = ['N','NE', 'E', 'SE','S','SW','W','NW']

def to_compass(heading, bearing_list) -> str:
    count = len(bearing_list)
    width = 360 / count
    half_width = width / 2
    return bearing_list[int(((heading + half_width) % 360) / width)]

def cpa(speed1,course1,speed2,course2,range,bearing):
    DTR = math.pi / 180
  
    x = range * math.cos(DTR*bearing)
    y = range * math.sin(DTR*bearing)
    xVel = speed2 * math.cos(DTR*course2) - speed1 * math.cos(DTR*course1)
    yVel = speed2 * math.sin(DTR*course2) - speed1 * math.sin(DTR*course1)
    dot = x * xVel + y * yVel
    if (dot >= 0.0):
        return
    a = xVel * xVel + yVel * yVel
    b = 2 * dot
    if (abs(a) < 0.0001 or abs(b) > 24 * abs(a)):
        return
    cpa = range * range - ((b*b)/(4*a))
    if (cpa <= 0.0):
        return (0, 60*(-b/(2*a)))
    cpa = math.sqrt(cpa)
    return (cpa, 60*(-b/(2*a)))


def distanceAfterSeconds(speedKts, timeSeconds):
  speedMs = speedKts * 0.51444444444444
  distanceM = speedMs * timeSeconds
  return distanceM


DEFAULT_SENSOR_TYPES = list(SENSOR_TYPES.keys())



PLATFORM_OPTIONS_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_POLL): cv.boolean,
        vol.Optional(CONF_SCAN_INTERVAL): cv.time_period,
        vol.Optional(CONF_USE_FAS_ICONS): cv.boolean,
        vol.Optional(CONF_SENSOR_TYPES): cv.ensure_list,
    },
    extra=vol.REMOVE_EXTRA,
)

SENSOR_SCHEMA = vol.Schema(
    {
        # TODO - work out how to use lower-case versions?
        #vol.Optional(CONF_DATA_SOURCE): cv.enum(DataSourceType),
        vol.Optional(CONF_DATA_SOURCE): cv.string,
        # TODO - work out how to make this conditional?
        vol.Optional(CONF_ADSB_SENSOR): cv.entity_id,
        vol.Optional(CONF_ADSB_JSON_ATTRIBUTE): cv.string,
        vol.Optional(CONF_ICON_TEMPLATE): cv.template,
        vol.Optional(CONF_FRIENDLY_NAME): cv.string,
        vol.Required(CONF_UNIQUE_ID): cv.string,
        vol.Optional(CONF_NAME): cv.string,
        vol.Optional(CONF_MQTT_TOPIC): cv.string,
    }
).extend(PLATFORM_OPTIONS_SCHEMA.schema)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_SENSORS): cv.schema_with_slug_keys(SENSOR_SCHEMA),
    }
).extend(PLATFORM_OPTIONS_SCHEMA.schema)


def compute_once_lock(sensor_type):
    """Only compute if sensor_type needs update, return just the value otherwise."""

    def wrapper(func):
        @wraps(func)
        async def wrapped(self, *args, **kwargs):
            async with self._compute_states[sensor_type].lock:
                if self._compute_states[sensor_type].needs_update:
                    setattr(self, f"_{sensor_type}", await func(self, *args, **kwargs))
                    self._compute_states[sensor_type].needs_update = False
                return getattr(self, f"_{sensor_type}", None)

        return wrapped

    return wrapper


async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    """Set up the ADSB sensors."""
    devices = discovery_info["devices"]
    options = discovery_info["options"]

    sensors = []

    for device_config in devices:
        device_config = options | device_config

        compute_device = DeviceAdsbInfo(
            hass=hass,
            name=device_config.get(CONF_NAME),
            unique_id=device_config.get(CONF_UNIQUE_ID),
            adsb_entity=device_config.get(CONF_ADSB_SENSOR),
            adsb_json_attribute=device_config.get(CONF_ADSB_JSON_ATTRIBUTE, JSON_ATTRIBUTE_DEFAULT),
            should_poll=device_config.get(CONF_POLL, POLL_DEFAULT),
            data_source=DataSourceType.from_string(device_config.get(CONF_DATA_SOURCE, DATA_SOURCE_DEFAULT)),
            mqtt_topic=device_config.get(CONF_MQTT_TOPIC, ''),
            scan_interval=device_config.get(
                CONF_SCAN_INTERVAL, timedelta(seconds=SCAN_INTERVAL_DEFAULT)
            ),
        )

        sensors += [
            SensorAdsbInfo(
                device=compute_device,
                entity_description_data=SENSOR_TYPES[SensorType.from_string(sensor_type)],
                icon_template=device_config.get(CONF_ICON_TEMPLATE),
                sensor_type=SensorType.from_string(sensor_type),
                fas_icons=device_config.get(CONF_USE_FAS_ICONS, False),
                is_config_entry=False,
            )
            for sensor_type in device_config.get(
                CONF_SENSOR_TYPES, DEFAULT_SENSOR_TYPES
            )
        ]

    async_add_entities(sensors)
    return True


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up entity configured via user interface.

    Called via async_forward_entry_setups(, SENSOR) from __init__.py
    """
    data = hass.data[DOMAIN][config_entry.entry_id]
    if data.get(CONF_SCAN_INTERVAL) is None:
        hass.data[DOMAIN][config_entry.entry_id][
            CONF_SCAN_INTERVAL
        ] = SCAN_INTERVAL_DEFAULT
        data[CONF_SCAN_INTERVAL] = SCAN_INTERVAL_DEFAULT

    _LOGGER.debug(f"async_setup_entry: {data}")
    compute_device = DeviceAdsbInfo(
        hass=hass,
        name=data[CONF_NAME],
        unique_id=f"{config_entry.unique_id}",
        adsb_entity=data[CONF_ADSB_SENSOR],
        adsb_json_attribute=data[CONF_ADSB_JSON_ATTRIBUTE],
        should_poll=data[CONF_POLL],
        data_source=DataSourceType.from_string(data[CONF_DATA_SOURCE]),
        mqtt_topic=data[CONF_MQTT_TOPIC],
        scan_interval=timedelta(
            seconds=data.get(CONF_SCAN_INTERVAL, SCAN_INTERVAL_DEFAULT)
        ),
    )

    entities: list[SensorAdsbInfo] = [
        SensorAdsbInfo(
            device=compute_device,
            entity_description_data=SENSOR_TYPES[sensor_type],
            sensor_type=sensor_type,
            fas_icons=data[CONF_USE_FAS_ICONS],
        )
        for sensor_type in SensorType
    ]
    if CONF_ENABLED_SENSORS in data:
        for entity in entities:
            if entity.entity_description.key not in data[CONF_ENABLED_SENSORS]:
                entity.entity_description.entity_registry_enabled_default = False

    if entities:
        async_add_entities(entities)


def id_generator(unique_id: str, sensor_type: str) -> str:
    """Generate id based on unique_id and sensor type.

    :param unique_id: str: common part of id for all entities, device unique_id, as a rule
    :param sensor_type: str: different part of id, sensor type, as s rule
    :returns: str: unique_id+sensor_type
    """
    return unique_id + sensor_type


class SensorAdsbInfo(SensorEntity):
    """Representation of a ADSB Info Sensor."""

    def __init__(
        self,
        device: "DeviceAdsbInfo",
        sensor_type: SensorType,
        entity_description_data: dict,
        icon_template: Template = None,
        fas_icons: bool = False,
        is_config_entry: bool = True,
    ) -> None:
        """Initialize the sensor."""
        self._device = device
        self._sensor_type = sensor_type
        desc_data = entity_description_data.copy()
        desc_data["has_entity_name"] = is_config_entry
        if not is_config_entry:
            desc_data["name"] = (
                f"{self._device.name} {desc_data['name']}"
            )
        if fas_icons:
            if desc_data["key"] in FAS_ENTITY_ICONS:
                desc_data["icon"] = FAS_ENTITY_ICONS[desc_data["key"]]
        self.entity_description = SensorEntityDescription(desc_data)
        self._icon_template = icon_template
        self._attr_native_value = None
        self._attr_extra_state_attributes = {}
        self._attr_unique_id = id_generator(self._device.unique_id, sensor_type)
        self._attr_should_poll = False
        self._available = False
        self._picture = None

    @property
    def device_info(self) -> dict[str, Any]:
        """Return device information."""
        return self._device.device_info

    @property
    def available(self) -> bool:
        """Return whether we're available."""
        return self._available

    @property
    def entity_picture(self) -> Optional[str]:
        return self._picture

    @property
    def extra_state_attributes(self):
        """Return the state attributes."""
        return dict(
            self._device.extra_state_attributes, **self._attr_extra_state_attributes
        )

    async def async_added_to_hass(self):
        """Register callbacks."""
        self._device.sensors.append(self)
        if self._icon_template is not None:
            self._icon_template.hass = self.hass
        if self._device.compute_states[self._sensor_type].needs_update:
            self.async_schedule_update_ha_state(True)

        if self._device._data_source == DataSourceType.MQTT:
            await self._device._register_mqtt_listener()

    async def async_update(self):
        """Update the state of the sensor."""
        value = await getattr(self._device, self._sensor_type)()
        if value is None:  # can happen during startup
            self._available = False
            return None

        if value == STATE_UNAVAILABLE:
            self._available = False
            return None

        self._available = True
        self._picture = None
        if type(value) == tuple and len(value) >= 2:
            self._attr_extra_state_attributes = value[1]
            #if self._sensor_type == SensorType.DEW_POINT_PERCEPTION:
            #    self._attr_extra_state_attributes[ATTR_DEW_POINT] = value[1]
            self._attr_native_value = value[0]
            if len(value) == 3:
                self._picture = value[2]
        else:
            self._attr_native_value = value

        if self._icon_template is not None:
            property_name = "_attr_icon"

            try:
                setattr(self, property_name, template.async_render())
            except TemplateError as ex:
                friendly_property_name = property_name[1:].replace("_", " ")
                if ex.args and ex.args[0].startswith(
                    "UndefinedError: 'None' has no attribute"
                ):
                    # Common during HA startup - so just a warning
                    _LOGGER.warning(
                        "Could not render %s template %s," " the state is unknown.",
                        friendly_property_name,
                        self.name,
                    )
                else:
                    try:
                        setattr(self, property_name, getattr(super(), property_name))
                    except AttributeError:
                        _LOGGER.error(
                            "Could not render %s template %s: %s",
                            friendly_property_name,
                            self.name,
                            ex,
                        )


@dataclass
class ComputeState:
    """ADSB Info Calculation State."""

    needs_update: bool = False
    lock: Lock = None


class DeviceAdsbInfo:
    """Representation of a ADSB Info Sensor."""

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        unique_id: str,
        adsb_entity: str,
        adsb_json_attribute: str,
        should_poll: bool,
        scan_interval: timedelta,
        data_source: DataSourceType,
        mqtt_topic: str,
    ):
        """Initialize the sensor."""
        self.hass = hass
        self._unique_id = unique_id
        self._device_info = DeviceInfo(
            identifiers={(DOMAIN, self.unique_id)},
            name=name,
            manufacturer=DEFAULT_NAME,
            model="Virtual Device",
        )
        self.extra_state_attributes = {}
        self._adsb_entity = adsb_entity
        self._adsb_json_attribute = adsb_json_attribute
        self._adsb_info = None
        self._should_poll = should_poll
        self.sensors = []
        self._compute_states = {
            sensor_type: ComputeState(lock=Lock())
            for sensor_type in SENSOR_TYPES.keys()
        }

        self._data_source = data_source
        self._mqtt_topic = mqtt_topic
        self._mqtt_registered = False
        self._mqtt_lock = Lock()

        if self._data_source == DataSourceType.SENSOR:
            async_track_state_change_event(
                self.hass, self._adsb_entity, self.adsb_state_listener
            )

            hass.async_create_task(
                self._new_adsb_state(hass.states.get(adsb_entity))
            )

            if self._should_poll:
                if scan_interval is None:
                    scan_interval = timedelta(seconds=SCAN_INTERVAL_DEFAULT)
                async_track_time_interval(
                    self.hass,
                    self.async_update_sensors,
                    scan_interval,
                )
        else:
            # MQTT, we'll set this up as our sensors are added to hass
            pass

        hass.async_create_task(self._set_version())


    async def _set_version(self):
        self._device_info["sw_version"] = (
            await async_get_custom_components(self.hass)
        )[DOMAIN].version.string

    async def adsb_state_listener(self, event):
        """Handle ADSB device state changes."""
        await self._new_adsb_state(event.data.get("new_state"))

    async def _new_adsb_state(self, state):
        if _is_valid_state(state):
            hass = self.hass
            info = state.attributes.get(self._adsb_json_attribute)
            # TODO - check it's valid?
            self._info_json = info
            await self.async_update()
        else:
            _LOGGER.info(f"ADSB info has an invalid value: {state}. Can't calculate new states. This is OK if during initial boot.")

    async def adsb_mqtt_listener(self, message):
        """Handle ADSB device mqtt_message."""
        payload = json.loads(message.payload)
        # TODO: definitely don't hardcode this!
        self._info_json = payload.get('nearest_aircraft')
        if not self._info_json:
            self._info_json = payload.get('aircraft')
        await self.async_update()

    # TODO - there is absolutely a better way to do this than this complicated callback mechanism
    async def _register_mqtt_listener(self):
        async with self._mqtt_lock:
            if self._mqtt_registered:
                return

            await mqtt.async_subscribe(
                hass=self.hass,
                topic=self._mqtt_topic,
                msg_callback=self.adsb_mqtt_listener,
                qos=1
            )
            self._mqtt_registered = True

    @compute_once_lock(SensorType.TRACKED_COUNT)
    async def tracked_count(self) -> Optional[Tuple[int, dict]]:
        # TODO: go through and unhardcode these
        if self._info is None:
            return 0
        return (len(self._info), {'raw': self._info})

    def closest(self, max_dist=99999999, max_age=99999999, exclude_ground=True):
        if not self._info:
            return None
        for craft in self._info:
            if not craft['computed_has_coords']:
                next
            if exclude_ground and craft.get('alt_baro') == 'ground':
                next
            if craft.get('computed_dist') > max_dist:
                next
            if craft.get('seen') > max_age:
                next
            return craft
        return None

    def closest_interesting(self):
        # TODO pull these from preferences
        return self.closest(max_age=60)

    # TODO: error handling
    @compute_once_lock(SensorType.CLOSEST_AIRCRAFT)
    async def closest_aircraft(self) -> Optional[Tuple[str, dict, str]]:
        closest = self.closest_interesting()
        if not closest:
            return None

        # TODO: don't hardcode r
        reg = closest.get('r')
        if reg:
            reg = reg.strip()

        flight = closest.get('flight')
        if flight:
            flight = flight.strip()

        if not reg:
            if not flight:
                reg = "[???]"
            else:
                reg = f"[{flight}]"

        image = closest.get('image')

        category = (str(closest.get('category') or "") + "XX")[:2]

        attrs = {
            'latitude': closest.get('lat'),
            'longitude': closest.get('lon'),
            'operator': closest.get('operator'),
            'owner': closest.get('owner'),
            'route': closest.get('route'),
            'flight_number': closest.get('flight'),
            'type': closest.get('desc'),
            'type_code': closest.get('t'),
            'category_code': category,
            'category': category,
        }
        return (reg, attrs, image)

    # TODO: error handling
    # TODO deduplicate
    @compute_once_lock(SensorType.CLOSEST_AIRCRAFT_GROUND_SPEED)
    async def closest_aircraft_ground_speed(self) -> Optional[float]:
        closest = self.closest_interesting()
        if not closest:
            return None
        speed = closest.get('gs')
        return speed

    @compute_once_lock(SensorType.CLOSEST_AIRCRAFT_HEADING)
    async def closest_aircraft_heading(self) -> Optional[Tuple[float, dict]]:
        closest = self.closest_interesting()
        if not closest:
            return None
        closest = self._info[0]
        heading = closest.get('track')
        if not heading:
            return None
        attrs = {
            'rate': closest.get('track_rate'),
            'compass': to_compass(heading, BEARINGS),
            'compass_fine': to_compass(heading, BEARINGS_FINE),
            'compass_crude': to_compass(heading, BEARINGS_CRUDE),
        }
        return (heading, attrs)

    @compute_once_lock(SensorType.CLOSEST_AIRCRAFT_BAROMETRIC_ALTITUDE)
    async def closest_aircraft_barometric_altitude(self) -> Optional[Tuple[float, dict]]:
        closest = self.closest_interesting()
        if not closest:
            return None
        alt = closest.get('alt_baro')
        if not alt:
            return None
        if alt == 'ground':
            alt = 0
        baro_rate = closest.get('baro_rate')
        if baro_rate is None:
            baro_rate = STATE_UNKNOWN
        else:
            baro_rate = round(baro_rate, 2)
        attrs = {
            # TODO - fall back to geom rate?
            'rate': baro_rate,
        }
        return (alt, attrs)

    @compute_once_lock(SensorType.CLOSEST_AIRCRAFT_DISTANCE)
    async def closest_aircraft_distance(self) -> Optional[float]:
        closest = self.closest_interesting()
        if not closest:
            return None
        return closest.get('computed_dist')

    @compute_once_lock(SensorType.CLOSEST_AIRCRAFT_BEARING)
    async def closest_aircraft_bearing(self) -> Optional[Tuple[float, dict]]:
        # TODO - use one in JSON if it's there?
        # TODO - allow specification of different origin
        closest = self.closest_interesting()
        if not closest:
            return None
        bearing = closest.get('computed_bearing')
        if not bearing:
            return None
        attrs = {
            'compass': to_compass(bearing, BEARINGS),
            'compass_fine': to_compass(bearing, BEARINGS_FINE),
            'compass_crude': to_compass(bearing, BEARINGS_CRUDE),
        }
        return (bearing, attrs)

    @compute_once_lock(SensorType.CLOSEST_AIRCRAFT_APPROACHING)
    async def closest_aircraft_approaching(self) -> Optional[str]:
        closest = self.closest_interesting()
        if not closest:
            return None
        track = closest.get('track')
        bearing = closest.get('computed_bearing')
        if not track or not bearing:
            return None

        relative = int(bearing - track) % 360
        if relative > 90 and relative < 270:
            return ApproachingState.APPROACHING
        return ApproachingState.RECEDING

    @compute_once_lock(SensorType.CLOSEST_AIRCRAFT_CPA)
    async def closest_aircraft_cpa(self) -> Optional[Tuple[float, dict]]:
        closest = self.closest_interesting()
        if not closest:
            return None

        lat = float(closest.get('lat') or 0.0)
        long = float(closest.get('lon') or 0.0)
        bearing = closest.get('computed_bearing')
        distance = closest.get('computed_bearing')
        track = closest.get('track')
        speed = closest.get('gs')
        if lat == 0.0 and long == 0.0:
            return None
        if not track or not speed or not bearing or not distance:
            return None
        distance_nm = distance / 1.852
        relative = int(bearing - track) % 360
        if relative < 90 or relative > 270:
            # It's receding
            return None

        cpa_result = cpa(0, 0, speed, track, distance_nm, bearing)

        if cpa_result is None:
            return None

        cpa_km = cpa_result[0] * 1.852
        cpa_secs = cpa_result[1] * 60
        climb = closest.get('baro_rate')
        alt = closest.get('alt_baro')
        if climb is None or alt is None:
            cpa_est_altitude = STATE_UNKNOWN
        else:
            cpa_est_altitude = round(alt + (cpa_result[1] * climb), 0)

        dist_to_cover = distanceAfterSeconds(speed, cpa_secs)
        cpa_point = gcc.point_given_start_and_bearing((long, lat), track, dist_to_cover, 'meters')
        cpa_lat = cpa_point[1]
        cpa_long = cpa_point[0]

        attrs = {
            'altitude_est': cpa_est_altitude,
            'time_est': round(cpa_secs, 1),
            'latitude': cpa_lat,
            'longitude': cpa_long,
        }

        return (round(cpa_km, 3), attrs)

    def preprocess_json(self, raw_json):
        processed = raw_json
        for craft in processed:
            craft['computed_has_coords'] = False
            # TODO - option to allow geo preprocessing?
            lat = craft.get('lat')
            long = craft.get('lon')
            if (lat and long):
                lat = float(lat)
                long = float(long)
                bearing = gcc.bearing_at_p1((self.hass.config.longitude, self.hass.config.latitude), (long, lat))
                craft['computed_has_coords'] = True
                craft['computed_dist'] = self.hass.config.distance(lat, long)
                craft['computed_bearing'] = bearing

        # TODO only if we're using geo
        processed.sort(key=lambda x: x.get('computed_dist') or 999999.9999)

        return processed

    async def async_update(self):
        """Update the state."""
        if self._info_json is not None:
            self._info = self.preprocess_json(self._info_json)
            for sensor_type in SENSOR_TYPES.keys():
                self._compute_states[sensor_type].needs_update = True
            if not self._should_poll:
                await self.async_update_sensors(True)

    async def async_update_sensors(self, force_refresh: bool = False) -> None:
        """Update the state of the sensors."""
        for sensor in self.sensors:
            sensor.async_schedule_update_ha_state(force_refresh)

    @property
    def compute_states(self) -> dict[SensorType, ComputeState]:
        """Compute states of configured sensors."""
        return self._compute_states

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return self._unique_id

    @property
    def device_info(self) -> dict:
        """Return the device info."""
        return self._device_info

    @property
    def name(self) -> str:
        """Return the name."""
        return self._device_info["name"]


def _is_valid_state(state) -> bool:
    if state is not None:
        if state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return True
    return False
