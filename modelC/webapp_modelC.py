import io
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T' # to disable Ctrl+C crashing python when having scipy.interpolate imported (disables Fortrun runtime library from intercepting Ctrl+C signal and lets it to the Python interpreter)
from openpyxl import Workbook
from shiny import ui, App, reactive, render
from shinywidgets import render_widget, output_widget
from ipyleaflet import Map, Marker, MarkerCluster, WidgetControl, FullScreenControl, ScaleControl
from ipywidgets import SelectionSlider, Play, VBox, jslink, Layout, HTML, Label # pip install ipywidgets==7.6.5, because version 8 has an issue with popups (https://stackoverflow.com/questions/75434737/shiny-for-python-using-add-layer-for-popus-from-ipyleaflet)
import numpy as np
import pandas as pd
import xarray as xr
# pip install scipy==1.13.1 would be optimal, since interp2d much faster than RegularGridInterpolator and supports cubic interpolation, even if deprecated.
# However, this old version doesn't exist as manylinux prebuilt wheel, which is necessary, since the Azure environment doesn't have a Fortran compiler necessary for building from source.
# Therefore use of RegularGridInterpolator instead.
from scipy.interpolate import CubicSpline, RegularGridInterpolator #, interp2d 
import base64
from ecmwf.opendata import Client
import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
import math
import joblib
import torch
import torch.nn as nn
import datetime
import matplotlib.dates as mdates
import asyncio



##### Do once when programme is executed #####

# Limit to Europe
lat_min, lat_max = 35, 72
lon_min, lon_max = -25, 45
center_lat = (lat_min + lat_max) / 2
center_lon = (lon_min + lon_max) / 2

# Selectable time zones
timezone_list = ["UTC-1", "UTC", "UTC+1", "UTC+2", "UTC+3"]

# Filter data for Europe and extract relevant columns
df = pd.read_parquet("data_hosting/The_Wind_Power.parquet") # 0.7 seconds when WPPs already regionally filtered and stored as parquet file. As unfiltered excel file it takes 11 seconds
#df = df.iloc[::100] # only every 100th wpp
ids = df['ID'].values
countries = df['Country'].values
project_names = df['Name'].values
lats_plants = df['Latitude'].values
lons_plants = df['Longitude'].values
manufacturers = df['Manufacturer'].values
turbine_types = df['Turbine'].replace(["nan", np.nan], "nan").values
hub_heights = df['Hub height'].values
numbers_of_turbines = df['Number of turbines'].replace(0, "nan").values
capacities = df['Total power'].values / 1e3 # kW to MW as the model has been trained on and the capacity scaler has been fitted
developers = df['Developer'].values
operators = df['Operator'].values
owners = df['Owner'].values
commissioning_dates = df['Commissioning date'].values
ages_months = df['Ages months'].values
commissioning_date_statuses = df['Commissioning date status'].values
hub_height_statuses = df['Hub height status'].values
links = df['Link'].values

country_set = sorted(set(countries))

hub_height_min = math.floor(0.9 * df['Hub height'].min())
hub_height_max = math.ceil(1.1 * df['Hub height'].max())
commissioning_years = df['Commissioning date'].str.split('/').str[0].astype(int)
min_year = commissioning_years.min()
max_year = commissioning_years.max()
min_capacity = capacities.min()
max_capacity = capacities.max()

# Model definition
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3366)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

encoders = joblib.load("modelC/parameters_deployment/encoders.pkl")
encoder = encoders[0] # encoders are identical for all lead times (checked previously)
known_turbine_types = encoder.categories_[0]
selectable_turbine_types = np.concatenate((known_turbine_types, np.array(["unknown for model", "nan"])))
unknown_turbine_types = set([turbine_type for turbine_type in turbine_types if turbine_type not in known_turbine_types])
unknown_turbine_types = sorted(list(unknown_turbine_types))

scalers = joblib.load("modelC/parameters_deployment/scalers.pkl")

input_sizes = joblib.load("modelC/parameters_deployment/input_sizes.pkl")
input_size = input_sizes[0] # input sizes are identical for all lead times (checked previously)
models = {}
model_state_dicts = torch.load("modelC/parameters_deployment/models.pth", weights_only=True)
for lead_time, model_state_dict in model_state_dicts.items():
    model_lead_time = MLP(input_size)
    model_lead_time.load_state_dict(model_state_dict)
    model_lead_time.eval()
    models[lead_time] = model_lead_time

# Construct the webapp
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Spatial Forecast",
        ui.div(
            ui.panel_well("🛠 Preparing wind forecast data and building interactive map..."),
            id="initial_loading_message"
        ),
        output_widget("map")
    ),
    ui.nav_panel(
        "Temporal Forecast",
        ui.row(
            ui.column(2,  # Left column for input fields
                ui.input_slider("lat", "Latitude", min=lat_min, max=lat_max, value=(lat_min + lat_max) / 2, step=0.01),
                ui.input_slider("lon", "Longitude", min=lon_min, max=lon_max, value=(lon_min + lon_max) / 2, step=0.01),
                ui.input_select("turbine_type", "Turbine Type", choices=selectable_turbine_types.tolist(), selected=known_turbine_types[0]),
                ui.div(id="unknown_turbine_container"),
                ui.input_slider("hub_height", "Hub Height (m)", min=hub_height_min, max=hub_height_max, value=(hub_height_min + hub_height_max) / 2, step=0.1),
                ui.input_slider("commissioning_date_year", "Commissioning Date (Year)", min=min_year, max=max_year, value=(min_year + max_year) / 2, step=1, sep=''),
                ui.input_slider("commissioning_date_month", "Commissioning Date (Month)", min=1, max=12, value=6, step=1),
                ui.input_slider("capacity", "Capacity (MW)", min=min_capacity, max=max_capacity, value=(min_capacity + max_capacity) / 2, step=0.01),
                ui.tags.br(),
                ui.input_file("upload_file", "Contribute data for this configuration", accept=[".xlsx"]),
                ui.tags.a(
                    "Download Example File",
                    href="/example_time_series.xlsx",
                    download="example_time_series.xlsx",
                    target="_blank",
                    rel="noopener noreferrer",
                    style="text-decoration: none; padding: 0.5em; display: inline-block;"
                )
            ),
            ui.column(10,  # Right column for output
                ui.panel_well(  # Panel to centre the content
                    ui.output_ui("output_summary"),
                    ui.tags.br(),
                    ui.output_plot("output_graph"),
                    ui.tags.br(),
                    ui.input_action_button("action_button", "Download Forecast"),
                ),
            ),
        ),
        value='customise_WPP'
    ),
    ui.nav_panel(
        "Settings and Documentation",
        ui.input_select(
            "selected_timezone",  # Unique ID for global timezone selector
            "Select Time Zone",
            choices=timezone_list,
            selected="UTC"
        ),
        # separate URL for SEO
        ui.tags.div(
            ui.tags.a("Documentation", href="/documentation.html", target="_blank", 
                    style="padding: 0.5em 1em; margin: 0.5em; background-color: #007BFF; color: white; border: none; border-radius: 5px; text-decoration: none; display: inline-block;"),
        ),
        ui.tags.div(
            ui.tags.a("Source code on GitHub", href="https://github.com/el-gif/Wind-Power-Forecast", target="_blank",
                    style="padding: 0.5em 1em; margin: 0.5em; background-color: #28A745; color: white; border: none; border-radius: 5px; text-decoration: none; display: inline-block;"),
        )
    ),
    ui.head_content(
        # for SEO
        ui.tags.head(
            ui.tags.meta(name="description", content="Wind energy forecasting tool with interactive maps and real-time data"),
            ui.tags.meta(name="keywords", content="wind power, wind energy, forecasting, renewable energy, Europe, real-time, interactive map"),
            ui.tags.meta(name="author", content="Alexander Peters"),
            ui.tags.meta(name="robots", content="index, follow")
        ),
        ui.tags.link(rel="icon", type="image/png", href="/WPP_icon2.png"), # image source: https://www.kroschke.com/windsack-set-inkl-korb-und-huelle-korbdurchmesser-650mm-laenge-3500mm--m-8509.html
        ui.tags.script("""
            Shiny.addCustomMessageHandler("download_file", function(message) {
                var link = document.createElement('a');
                link.href = 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,' + message.data;
                link.download = message.filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            });
        """), # to download forecast and example Excel files
        ui.tags.script("""
            Shiny.addCustomMessageHandler("remove_element", function(id) {
                const el = document.getElementById(id);
                if (el) el.remove();
            });
        """) # to remove the "Preparing data" message at the beginning
    ),
    id="navbar_selected",
    title="Wind Power Forecast"
)

# Server function
def server(input, output, session):

    ##### Upon start of a new session #####

    if os.getenv("RENDER") or os.getenv("WEBSITE_HOSTNAME"):  # for Render or Azure Server
        root = "/home/data"
    else:
        root = "data"

    # Initialise ECMWF client
    client = Client()
    overwrite = 0  # force overwriting of data

    # Determine the latest available forecast based on the ECMWF dissemination schedule https://confluence.ecmwf.int/display/DAC/Dissemination+schedule
    current_utc = datetime.datetime.now(datetime.timezone.utc)

    if current_utc.hour >= 21:
        latest_time = 12  # 12 UTC run is available after 18:27 UTC + 1 hour (+ margin)
        latest_date = current_utc.date()
    elif current_utc.hour >= 9:
        latest_time = 0  # 00 UTC run is available after 06:27 UTC + 1 hour (+ margin)
        latest_date = current_utc.date()
    else:
        latest_time = 12
        latest_date = (current_utc - datetime.timedelta(days=1)).date()

    print(f"Latest available forecast run: {latest_date}, {latest_time} UTC")

    save_dir = os.path.join(root, "weather_forecast")
    os.makedirs(save_dir, exist_ok=True)
    new_file = f"forecast_{latest_date}_{latest_time}.grib"
    new_file_path = os.path.join(save_dir, new_file)

    # Check if the new forecast file is already available
    if os.path.exists(new_file_path) and overwrite == 0:
        print(f"Latest forecast file {new_file} is already available. No download needed.")
    else:
        for old_file in os.listdir(save_dir):
            if old_file.startswith("forecast"):  # Filter files
                old_file_path = os.path.join(save_dir, old_file)  # Get full path
                if os.path.isfile(old_file_path):  # Ensure it's a file (not a folder)
                    print(f"Deleting old file: {old_file}")
                    os.remove(old_file_path)
                    print("File deleted.")

        # Download the new forecast file
        print(f"Downloading new forecast file {new_file}")
        
        # Fetch the latest ECMWF forecast data
        result = client.retrieve(
            type="fc",
            param=["100v", "100u"],  # U- and V-components of wind speed
            target=new_file_path,
            time=latest_time,  # Use the latest available forecast run
            step=list(range(0, 145, 3))
        )

        print(f"New forecast file {new_file} successfully downloaded.")

    # Load the wind data (Grib2 file)
    ds = xr.open_dataset(new_file_path, engine='cfgrib')
    lats_world = ds['latitude'].values
    lons_world = ds['longitude'].values
    u_world = ds['u100'].values
    v_world = ds['v100'].values
    valid_times = ds['valid_time'].values

    # Filter for Europe
    lat_indices = np.where((lats_world >= lat_min) & (lats_world <= lat_max))[0]
    lon_indices = np.where((lons_world >= lon_min) & (lons_world <= lon_max))[0]
    lats = lats_world[lat_indices]  # actual latitude values
    lons = lons_world[lon_indices]  # actual longitude values
    u = u_world[:, lat_indices[:, None], lon_indices]  # Shape: (time, lat_subset, lon_subset)
    v = v_world[:, lat_indices[:, None], lon_indices]

    # Calculate total wind speed and convert to 3D array
    total_selection = np.array([np.sqrt(u_value**2 + v_value**2) for u_value, v_value in zip(u, v)])

    start_time = valid_times[0]
    end_time = valid_times[-1]
    total_hours = int((end_time - start_time) / np.timedelta64(1, 'h'))
    step_size = np.timedelta64(3, 'h') # imposed by ECMWF open weather forecast
    step_size_hours = int(step_size / np.timedelta64(1, 'h'))


    ##### Page 1 #####

    # Define reactive values
    project_name = reactive.Value(None)
    operator = reactive.Value(None)
    owner = reactive.Value(None)
    commissioning_date_status = reactive.Value(None)
    hub_height_status = reactive.Value(None)
    country = reactive.Value(None)
    number_turbines = reactive.Value(None)

    is_programmatic_change = reactive.Value(False)

    @reactive.effect
    @reactive.event(input.entire_forecast)
    def entire_forecast_function():

        id = input.entire_forecast()['id']
        time_series.set(None)
        
        # Setzen des Flags: Änderungen sind programmatisch
        is_programmatic_change.set(True)

        index = list(ids).index(id)

        # Speichern der zusätzlichen Informationen in den reaktiven Werten
        project_name.set(project_names[index])
        operator.set(operators[index])
        owner.set(owners[index])
        commissioning_date_status.set(commissioning_date_statuses[index])
        hub_height_status.set(hub_height_statuses[index])
        country.set(countries[index])
        number_turbines.set(numbers_of_turbines[index])

        # Parameter extrahieren
        lat = lats_plants[index]
        lon = lons_plants[index]
        turbine_type = turbine_types[index]
        hub_height = hub_heights[index]
        capacity = capacities[index]
        commissioning_date = commissioning_dates[index]
        commissioning_date_year, commissioning_date_month = commissioning_date.split("/")
        commissioning_date_year = int(commissioning_date_year)
        commissioning_date_month = int(commissioning_date_month)

        # Update der Eingabefelder mit neuen Werten vor Wechseln des Tabs.
        # Attention: the rounding must exactly correspond to the steps, the sliders have been initialised with.
        # Otherwise, a rounding will occur automatically, and the function observe_slider_changes() is called an additional time,
        # while is_programmatic_change is already false --> output_summary will be reset --> to avoid
        ui.update_slider("lat", value=round(lat, 2))
        ui.update_slider("lon", value=round(lon, 2))
        ui.update_select("turbine_type", selected=turbine_type if turbine_type in known_turbine_types or turbine_type == "nan" else "unknown for model")
        ui.update_slider("hub_height", value=round(hub_height, 1))
        ui.update_slider("commissioning_date_year", value=commissioning_date_year)
        ui.update_slider("commissioning_date_month", value=commissioning_date_month)
        ui.update_slider("capacity", value=(capacity, 2))

        # Wechsel zu "Customise WPP" Tab
        ui.update_navs("navbar_selected", selected="customise_WPP")


    @render_widget
    async def map():

        await session.send_custom_message("remove_element", "initial_loading_message")
        marker_storage = {"marker": []}

        # Get the user-selected timezone directly
        timezone_str = input.selected_timezone()
        if timezone_str is None: # can happen during startup due to reactive environment
            timezone_offset = 0
        else:
            timezone_offset = int(timezone_str.replace("UTC", "").lstrip("+") or 0)

        # Apply the time shift from UTc to local time
        time_shift = np.timedelta64(timezone_offset, 'h')
        valid_times_local = valid_times + time_shift  # Update timestamps

        # Create the map
        m = Map(
            center=[center_lat, center_lon],
            zoom=11,
            layout=Layout(width='100%', height='90vh'),
            scroll_wheel_zoom=True
        )
        update_enabled = {"value": True}  # mutable flag
        
        scale = ScaleControl(position="bottomleft", imperial=False)
        m.add(scale)

        # Slider for time steps
        play = Play(min=0, max=total_hours, step=1, value=0, interval=2000, description='Time Step') # 2000 ms per time step
        valid_times_dt = pd.to_datetime(valid_times_local)
        formatted_times = [t.strftime('%d/%m %H:%M') for t in valid_times_dt] # known bug: SelectionSlider gives too little place for values --> year can't be displayed
        slider = SelectionSlider(options=formatted_times, value=formatted_times[0], description='Time')
        jslink((play, 'value'), (slider, 'index'))
        slider_box = VBox([play, slider], layout=Layout(object_position="center"))
        m.add(WidgetControl(widget=slider_box, position='topright'))

        # Label for zoom level --> too slow, particularly when markers are added, although the code snippet is above the marker adding
        zoom_label = Label(value=f"Zoom: {m.zoom}", layout=Layout(width='80px'))
        m.add(WidgetControl(widget=zoom_label, position='bottomright'))

        # Update when zoom changes
        def update_zoom_label(event):
            zoom_label.value = f"Zoom: {m.zoom:.1f}"

        m.observe(update_zoom_label, names=["zoom"])

        def on_marker_click(event=None, **kwargs):
            update_enabled["value"] = False  # disable zoom update when just a popup menu is clicked

        async def handle_zoom_change(event):

            if not update_enabled["value"]:
                print("Zoom handler disabled.")
                update_enabled["value"] = True #re-enable zoom update
                return

           
            zoom_level = m.zoom
            print(f"Current zoom: {zoom_level}")

            # Remove existing marker cluster if present
            for layer in m.layers:
                if isinstance(layer, MarkerCluster):
                    m.remove(layer)

            if zoom_level < 11:
                return
            
            await asyncio.sleep(0.1)  # wait 100 ms before getting m.bounds to ensure they are updated

            # initially, bound are not yet set
            if not m.bounds or len(m.bounds) != 2:
                south_west = (center_lat - 0.2, center_lon - 0.4)
                north_east = (center_lat + 0.2, center_lon + 0.4)
            else:
                south_west, north_east = m.bounds

            lat_min_visible, lon_min_visible = south_west
            lat_max_visible, lon_max_visible = north_east

            filtered_indices = [
                i for i in range(len(lats_plants))
                if lat_min_visible <= lats_plants[i] <= lat_max_visible
                and lon_min_visible <= lons_plants[i] <= lon_max_visible
            ]

            # Optional: Store somewhere or update the map accordingly
            filtered_data = pd.DataFrame({
                "lats": [lats_plants[i] for i in filtered_indices],
                "lons": [lons_plants[i] for i in filtered_indices],
                "ids": [ids[i] for i in filtered_indices],
                "names": [project_names[i] for i in filtered_indices],
                "capacities": [capacities[i] for i in filtered_indices],
                "numbers": [numbers_of_turbines[i] for i in filtered_indices],
                "turbines": [turbine_types[i] for i in filtered_indices],
                "operators": [operators[i] for i in filtered_indices],
                "ages": [ages_months[i] for i in filtered_indices],
                "hub_heights": [hub_heights[i] for i in filtered_indices],
                "links": [links[i] for i in filtered_indices]
            })

            marker_storage["marker"] = []
            for _, row in filtered_data.iterrows():                
                popup_content = HTML(
                    f"<strong>Project Name:</strong> {row['names']}<br>"
                    f"<strong>Capacity:</strong> {row['capacities']} MW<br>"
                    f"<strong>Number of turbines:</strong> {row['numbers']}<br>"
                    f"<strong>Turbine Type:</strong> {row['turbines']}<br>"
                    f"<strong>Operator:</strong> {row['operators']}<br>"
                    f"<strong>Wind speed forecast:</strong> select forecast step<br>"
                    f"<strong>Production forecast:</strong> select forecast step<br>"
                    f"<strong><a href='{row['links']}' target='_blank' style='color:blue; text-decoration:underline;'>Link to The Wind Power</a></strong><br>"
                    f"<button onclick=\"Shiny.setInputValue('entire_forecast', {{id: {row['ids']}, timestamp: Date.now()}})\">Entire Forecast</button>"
                )

                marker = Marker(
                    location=(row["lats"], row["lons"]),
                    popup=popup_content,
                    rise_offset=True,
                    draggable=False
                )

                marker.on_click(on_marker_click)
                marker_storage["marker"].append(marker)

            marker_cluster = MarkerCluster(markers=marker_storage["marker"])
            m.add(marker_cluster)

            def update_predictions(change):

                if filtered_data["ids"].empty:
                    return

                # Convert the selected slider value (HH:MM dd/mm) back to datetime
                time_step_local = pd.to_datetime(slider.value, format='%d/%m %H:%M')  # Convert to datetime

                # Get the current year in UTC
                current_utc = datetime.datetime.now(datetime.timezone.utc)
                time_shift = datetime.timedelta(hours=timezone_offset)
                current_local = current_utc + time_shift
                current_year = current_local.year

                # If the selected month is January and the current month is December, set the year to next year
                if time_step_local.month == 1 and current_local.month == 12:
                    corrected_year = current_year + 1
                else:
                    corrected_year = current_year  # Otherwise, keep the current year

                # Reconstruct the full datetime with the correct year
                time_step_local = time_step_local.replace(year=corrected_year)

                # convert to UTC
                time_step = time_step_local - time_shift
                lead_time = int((time_step - start_time) / np.timedelta64(1, 'h'))
                step_index = int(lead_time / step_size_hours)
                    
                # total_selection[step_index] should have shape (len(lats), len(lons))
                spatial_interpolator = RegularGridInterpolator(
                    (lats, lons),  # note: order is (y, x)
                    total_selection[step_index], 
                    method='linear', 
                    bounds_error=False, 
                    fill_value=None
                )

                # Each point must be (lat, lon)
                wind_speeds_at_points = spatial_interpolator(np.column_stack((filtered_data["lats"], filtered_data["lons"])))

                # scaling
                scaled_ages_months = scalers[lead_time]["ages"].transform(np.array(filtered_data["ages"]).reshape(-1, 1)).flatten()
                scaled_hub_heights = scalers[lead_time]["hub_heights"].transform(np.array(filtered_data["hub_heights"]).reshape(-1, 1)).flatten()
                scaled_wind_speeds_at_points = scalers[lead_time]["winds"].transform(wind_speeds_at_points.reshape(-1, 1)).flatten()

                number_wpps = len(filtered_data["ids"])
                turbine_types_onehot = np.zeros((number_wpps, len(known_turbine_types)))
                for i, turbine_type in enumerate(filtered_data["turbines"]):
                    if turbine_type not in known_turbine_types:
                        turbine_types_onehot[i] = np.full(len(known_turbine_types), 1.0 / len(known_turbine_types)) # equal mixture of all known turbine types
                    else:
                        turbine_types_onehot[i] = encoder.transform(np.array([[turbine_type]])).flatten()

                all_input_features = np.hstack([
                    turbine_types_onehot,
                    scaled_hub_heights.reshape(-1, 1),
                    scaled_ages_months.reshape(-1, 1),
                    scaled_wind_speeds_at_points.reshape(-1, 1)
                ])

                input_tensor = torch.tensor(all_input_features, dtype=torch.float32)

                model = models[lead_time]
                with torch.no_grad():
                    cap_factors = torch.clamp(model(input_tensor).flatten(), min=0.0, max=1.0)
                    predictions = cap_factors * torch.tensor(filtered_data["capacities"], dtype=torch.float32)

                predictions = predictions.numpy()
                
                # Update marker pop-ups with production values
                for marker, (_, row), wind_speed, prediction in zip(
                    marker_storage["marker"],
                    filtered_data.iterrows(),
                    wind_speeds_at_points,
                    predictions
                ):
                    marker.popup.value = \
                        f"<strong>Project Name:</strong> {row['names']}<br>"\
                        f"<strong>Capacity:</strong> {row['capacities']} MW<br>"\
                        f"<strong>Number of Turbines:</strong> {row['numbers']}<br>"\
                        f"<strong>Turbine Type:</strong> {row['turbines']}<br>"\
                        f"<strong>Operator:</strong> {row['operators']}<br>"\
                        f"<strong>Wind speed forecast:</strong> {wind_speed:.2f} m/s<br>"\
                        f"<strong>Production forecast:</strong> {prediction:.2f} MW<br>"\
                        f"<strong><a href='{row['links']}' target='_blank' style='color:blue; text-decoration:underline;'>Link to The Wind Power</a></strong><br>"\
                        f"<button onclick=\"Shiny.setInputValue('entire_forecast', {{id: {row['ids']}, timestamp: Date.now()}})\">Entire Forecast</button>" # timestamp to always have a slightly different button value to ensure that each and every click on the "entire forecast" button triggers the event, even click on same button twice

            update_predictions(None)
            slider.observe(update_predictions, names='value')

        await handle_zoom_change(None)
        m.observe(lambda e: asyncio.create_task(handle_zoom_change(e)), names=["zoom", "center"])
#        m.observe(handle_zoom_change, names=["zoom", "center"])

        # Add FullScreenControl to map
        m.add(FullScreenControl())

        return m
    

    ##### Page 2 #####

    forecast_data = reactive.Value({"wind_speeds": None, "productions": None})
    time_series = reactive.Value(None)
    button_status = reactive.Value("download")

    # Function to handle file upload
    @reactive.effect
    @reactive.event(input.upload_file)
    def handle_file_upload():
        if input.upload_file() is not None:
            try:
                file = input.upload_file()[0]['datapath']

                # Check if the file is empty
                if os.path.getsize(file) == 0:
                    ui.notification_show("File is empty. Please check the file and try again.", duration=None)
                    return

                # Read first sheet only (ignore sheet name entirely)
                time_series_data = pd.read_excel(file, sheet_name=0)

                # Check if DataFrame is empty
                if time_series_data.empty:
                    ui.notification_show("The Excel file contains no data. Please provide a valid time series.", duration=None)
                    return

                # Check if required columns are present
                required_columns = ["Date", "Production (MW)"]
                if not all(col in time_series_data.columns for col in required_columns):
                    ui.notification_show(f"The first sheet of the Excel file is missing required columns. Please ensure it includes {required_columns}.", duration=None)
                    return

                # Check if all dates are between 01/01/2000 and current date
                date_min = pd.Timestamp("2000-01-01")
                date_max = pd.Timestamp.now()
                if not time_series_data["Date"].between(date_min, date_max).all():
                    ui.notification_show("The time series contains dates outside 01/01/2000 and current date, which is inadmissible.", duration=None)
                    return
                
                time_series_data = time_series_data[required_columns]
                time_series.set(time_series_data)  # Calls output_graph function
                ui.update_action_button("action_button", label="Contribute Data")
                button_status.set('contribute')

            except FileNotFoundError:
                ui.notification_show("File not found. Please upload a valid file.", duration=None)

            except ValueError as ve:
                ui.notification_show(f"Value error: {str(ve)}. Ensure the file format is correct.", duration=None)

            except pd.errors.ExcelFileError:
                ui.notification_show("The file format is not recognised as an Excel file. Please upload a valid .xlsx or .xls file.", duration=None)

            except Exception as e:
                # Generic catch-all for any other errors
                ui.notification_show(f"An unexpected error occurred: {str(e)}. Please try again.", duration=None)

    # Observing slider changes to revert to forecast view
    @reactive.effect
    @reactive.event(input.lat, input.lon, input.turbine_type, input.hub_height, input.commissioning_date_year, input.commissioning_date_month, input.capacity)
    def observe_slider_changes():
        # Überspringen, wenn Änderungen programmatisch sind
        if is_programmatic_change.get():
            is_programmatic_change.set(None)
            return

        # reset reactive variables
        project_name.set(None)
        operator.set(None)
        owner.set(None)
        commissioning_date_status.set(None)
        hub_height_status.set(None)
        country.set(None)
        number_turbines.set(None)

        # display regular elements
        ui.update_action_button("action_button", label="Download Forecast")
        button_status.set('download')

    # Capture user input and display configuration summary
    @render.text
    def output_summary():
        # Capture inputs
        lat_plant = input.lat()
        lon_plant = input.lon()
        turbine_type = input.turbine_type()
        hub_height = input.hub_height()
        commissioning_date_year = input.commissioning_date_year()
        commissioning_date_month = input.commissioning_date_month()
        capacity = input.capacity()

        if turbine_type == "unknown for model" or turbine_type == "nan":
            turbine_type_display = "average turbine type: linear combination of all known turbine types (representative value)"
        else:
            turbine_type_display = turbine_type

        if hub_height_status.get() == 1:
            hub_height_addition = "(not verified, representative value)"
        else: # hub_height_status.get() == 0 or None (custom configuration)
            hub_height_addition = ""

        if commissioning_date_status.get() == 2:
            date_addition = "(not verified, representative value)"
        elif commissioning_date_status.get() == 1:
            date_addition = "(month not verified, representative value)"
        else: # commissioning_date_status.get() == 0 or None (custom configuration)
            date_addition = ""

        # Return configuration summary text
        summary_html = (
            f"<b>Wind Power Plant Configuration</b><br><br>"
            f"<b>Project Name:</b> {project_name.get()}<br>"
            f"<b>Country:</b> {country.get()}<br>"
            f"<b>Capacity:</b> {capacity} MW<br>"
            f"<b>Number of Turbines:</b> {number_turbines.get()}<br>"
            f"<b>Operator:</b> {operator.get()}<br>"
            f"<b>Owner:</b> {owner.get()}<br>"
            f"<b>Location:</b> ({lat_plant:.2f}, {lon_plant:.2f})<br>"
            f"<b>Turbine Type:</b> {turbine_type_display}<br>"
            f"<b>Hub Height:</b> {hub_height:.2f} m {hub_height_addition}<br>"
            f"<b>Commissioning Date:</b> {commissioning_date_month}/{commissioning_date_year} {date_addition}<br>"

        )
        return ui.HTML(summary_html)
    
    # Dynamically insert/remove the additional turbine type input field
    @reactive.effect
    @reactive.event(input.turbine_type)
    def toggle_unknown_turbine():
        ui.remove_ui("#unknown_turbine_wrapper")  # Remove the wrapper div
        if input.turbine_type() == "unknown for model" or input.turbine_type() == "nan":
            ui.insert_ui(
                ui.div(
                    ui.input_select("unknown_turbine", "Select Turbine Type for Crowdsourcing", 
                                    choices=unknown_turbine_types,
                                    selected=unknown_turbine_types[0]),
                    id="unknown_turbine_wrapper"  # Wrapper div
                ),
                selector="#unknown_turbine_container"
            )            


    # Plot forecasted production over time
    @render.plot
    def output_graph():
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Production (MW)")

        # Add horizontal dashed line for capacity
        capacity = input.capacity()
        ax1.axhline(y=capacity, color='gray', linestyle='--', label=f"Capacity ({capacity} MW)")

        time_series_data = time_series.get()

        if time_series_data is not None and not time_series_data.empty: # check if user has uploaded time series to plot it. In this case no consideration of time zone set by user in the corresponding tab, assuming that the uploaded time series contains time stamps from their own time zone. Conversion to UTC only when downloading the crowdsourced data to the server 
            if len(time_series_data.iloc[:, 0]) < 4: # CubicSpline needs at least 4 points to interpolate meaningfully
                ui.notification_show("Too few data points to plot")
                fig, ax1 = plt.subplots(figsize=(8, 4))
                return fig

            ax1.set_title("Historical Production")

            # Convert time values to numeric format for interpolation
            time_numeric = mdates.date2num(time_series_data.iloc[:, 0])
            cs = CubicSpline(time_numeric, time_series_data.iloc[:, 1])

            # Create a fine time grid for the smooth curve
            fine_time_grid = np.linspace(time_numeric[0], time_numeric[-1], 500)
            smooth_production = np.maximum(cs(fine_time_grid), 0).flatten() # cut off at 0 but not at capacity in case the time series is erroneous

            # Plot the smoothed curve
            ax1.plot(mdates.num2date(fine_time_grid), smooth_production, label="Smoothed (CubicSpline)", color="tab:orange", linestyle="--")

            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))

            ax2 = ax1.twinx()
            y_min, y_max = ax1.get_ylim()
            ax2.set_ylim(y_min / capacity, y_max / capacity)
            ax2.set_ylabel("Production (per unit)")

            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right') # Rotate the labels for better fit

            ax1.grid(True)
            plt.tight_layout()
            return fig
        
        else: # Retrieve inputs
            lat_plant = input.lat()
            lon_plant = input.lon()
            turbine_type = input.turbine_type()
            hub_height = input.hub_height()
            commissioning_date_year = input.commissioning_date_year()
            commissioning_date_month = input.commissioning_date_month()
            ref_date = pd.Timestamp("2024-12-01")
            age_months = (ref_date.year - commissioning_date_year) * 12 + (ref_date.month - commissioning_date_month)
            capacity = input.capacity()

            # Calculate forecasted productions
            predictions_power = []
            wind_speeds_at_point = []
            num_steps = len(valid_times)
            for step_index in range(num_steps):
                lead_time = step_index * step_size_hours
                model = models[lead_time]

                # total_selection[step_index] should have shape (len(lats), len(lons))
                spatial_interpolator = RegularGridInterpolator(
                    (lats, lons),  # note: order is (y, x)
                    total_selection[step_index], 
                    method='linear', 
                    bounds_error=False, 
                    fill_value=None
                )
                wind_speeds_at_point.append(spatial_interpolator((lat_plant, lon_plant)))
                # spatial_interpolator = interp2d(lons, lats, total_selection[step_index], kind='cubic')
                # wind_speeds_at_point.append(spatial_interpolator(lon_plant, lat_plant)[0])

                # scaling
                scaled_hub_height = scalers[lead_time]["hub_heights"].transform(np.array([[hub_height]]))[0][0]
                scaled_age_months = scalers[lead_time]["ages"].transform(np.array([[age_months]]))[0][0]
                scaled_wind_speed_at_point = scalers[lead_time]["winds"].transform(np.array([[wind_speeds_at_point[-1]]]))[0][0]

                if turbine_type == "unknown for model" or turbine_type == "nan":
                    num_categories = len(known_turbine_types)
                    one_hot_vector = np.full((1, num_categories), 1 / num_categories) # mixture of all known turbine types
                else:
                    one_hot_vector = encoder.transform(np.array([[turbine_type]]))

                all_input_features = np.hstack([
                    one_hot_vector, # Shape (1, num_categories)
                    np.array([[scaled_hub_height]]), # Convert scalar to (1, 1)
                    np.array([[scaled_age_months]]), # Convert scalar to (1, 1)
                    np.array([[scaled_wind_speed_at_point]]) # Convert scalar to (1, 1)
                ])

                input_tensor = torch.tensor(all_input_features, dtype=torch.float32)

                with torch.no_grad():
                    predictions_cap_factor = torch.clamp(model(input_tensor), min=0.0, max=1.0)

                predictions_power.append(predictions_cap_factor * capacity)

            predictions_power = np.array([p.item() for p in predictions_power]).reshape(-1, 1)  # Shape (49, 1)

            # Store the forecast data in the reactive `forecast_data`
            forecast_data.set({"wind_speeds": wind_speeds_at_point, "productions": predictions_power.flatten()})

            # Convert valid_times to pandas DatetimeIndex (assuming it is in UTC)
            valid_times_utc = pd.to_datetime(valid_times, utc=True)

            # Convert user's selected timezone (e.g., "UTC-1") to an integer offset
            timezone_str = input.selected_timezone()
            if timezone_str is None: # can happen during startup due to reactive environment
                timezone_offset = 0
            else:
                timezone_offset = int(timezone_str.replace("UTC", "").lstrip("+") or 0)

            # Apply time shift (convert offset to timedelta)
            time_shift = np.timedelta64(timezone_offset, 'h')
            valid_times_local = valid_times_utc + time_shift  # Shifted time for plotting
            valid_times_local = valid_times_local.to_numpy(dtype="datetime64[ns]")

            # Generate a fine time grid in local time
            fine_time_grid_local = pd.date_range(start=valid_times_local[0], end=valid_times_local[-1], periods=500)

            # Use numeric values in CubicSpline
            cs = CubicSpline(valid_times_local, predictions_power)

            # Get interpolated values
            smoothed_predictions = np.maximum(cs(fine_time_grid_local), 0).flatten()
            smoothed_predictions = np.minimum(smoothed_predictions, capacity).flatten()

            # Adjust the current time (from system UTC to user-selected timezone)
            current_time_local = pd.Timestamp.now(tz="UTC") + time_shift  # Shift current time

            # Update the plot with the new time axis
            ax1.plot(fine_time_grid_local, smoothed_predictions, '-', label="Prediction", color="blue")
            ax1.axvline(x=current_time_local, color='red', linestyle='--', label='Current Time', linewidth=1.5)

            # Plot original and smoothed data
            #ax1.plot(valid_times_local, predictions_power, 'o', label="Original Points", color="red")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Production (MW)")
            ax1.set_ylim(bottom=0)
            ax1.set_title("Forecasted Production")

            ax2 = ax1.twinx()
            y_min, y_max = ax1.get_ylim()
            ax2.set_ylim(y_min / capacity, y_max / capacity)
            ax2.set_ylabel("Production (per unit)")

            ax1.xaxis.set_major_locator(mdates.DayLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))

            # Rotate the labels for better fit
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

            ax1.grid(True)
            plt.tight_layout()
            return fig

    def download_forecast_data():

        data = forecast_data.get()
        wind_speeds, productions = data["wind_speeds"], data["productions"]
        
        lat_plant = input.lat()
        lon_plant = input.lon()
        turbine_type = input.turbine_type()
        hub_height = input.hub_height()
        commissioning_date_year = input.commissioning_date_year()
        commissioning_date_month = input.commissioning_date_month()
        capacity = input.capacity()

        # Create a new workbook and add data
        wb = Workbook()
        
        # Sheet1: Production Forecast
        ws_forecast = wb.active
        ws_forecast.title = "Production Forecast"

        # Add headers for date, wind speed, and production
        ws_forecast.append(["Date", "Wind Speed (m/s)", "Production (MW)"])

        # Extract user's timezone offset (e.g., "UTC-3" → -3)
        timezone_str = input.selected_timezone()
        if timezone_str is None: # can happen during startup due to reactive environment
            timezone_offset = 0
        else:
            timezone_offset = int(timezone_str.replace("UTC", "").lstrip("+") or 0)

        # Convert user-local timestamps back to UTC
        time_shift = np.timedelta64(timezone_offset, 'h')
        valid_times_local = valid_times + time_shift

        # Add production data per time step
        for time, wind_speed, production in zip(valid_times_local, wind_speeds, productions):
            time_as_datetime = pd.to_datetime(time).to_pydatetime()
            ws_forecast.append([time_as_datetime, float(wind_speed), float(production)])  # Ensure wind speed and production values are correctly added

        # Sheet2: Turbine Specifications
        ws_specs = ws_specs = wb.create_sheet("Turbine Specifications")

        if hub_height_status.get() == 1:
            hub_height_addition = "(not verified, representative value)"
        else: # hub_height_status.get() == 0 or None (custom configuration)
            hub_height_addition = ""

        if commissioning_date_status.get() == 2:
            date_addition = "(not verified, representative value)"
        elif commissioning_date_status.get() == 1:
            date_addition = "(month not verified, representative value)"
        else: # commissioning_date_status.get() == 0 or None (custom configuration)
            date_addition = ""

        # Define the specs_data list with required values only
        specs_data = [
            ["Specification", "Value"],
            ["Project Name", project_name.get()],
            ["Country", country.get()],
            ["Capacity (MW)", capacity],
            ["Number of Turbines", number_turbines.get()],
            ["Operator", operator.get()],
            ["Owner", owner.get()],
            ["Location", f"({lat_plant}, {lon_plant})"],
            ["Type", turbine_type],
            ["Hub Height", f"{hub_height:.2f} m {hub_height_addition}"],
            ["Commissioning Date", f"{commissioning_date_month}/{commissioning_date_year} {date_addition}"]
        ]

        # Populate the workbook
        for row in specs_data:
            ws_specs.append(row)

        # Save in buffer and return
        buffer = io.BytesIO()
        wb.save(buffer)
        buffer.seek(0)
        return buffer.getvalue()

    # Function to trigger download on button click
    @reactive.effect
    @reactive.event(input.action_button)
    async def action():
        if button_status.get() == "download":
        
            # Retrieve forecast data as bytes
            file_data = download_forecast_data()
            
            # Encode file in Base64
            file_data_base64 = base64.b64encode(file_data).decode('utf-8')
            
            # Send file as Base64 to client for download
            await session.send_custom_message("download_file", {
                "data": file_data_base64,
                "filename": "forecasted_production.xlsx"
            })
        
        else: # button_status.get() == "contribute"
            time_series_data = time_series.get()

            timezone_str = input.selected_timezone()
            if timezone_str is None: # can happen during startup due to reactive environment
                timezone_offset = 0
            else:
                timezone_offset = int(timezone_str.replace("UTC", "").lstrip("+") or 0)

            # Convert user-local timestamps back to UTC
            time_shift = np.timedelta64(timezone_offset, 'h')
            time_series_data["Date"] = time_series_data["Date"] - time_shift

            # Notify the user that the file is being saved
            ui.notification_show(
                "Thank you for uploading your time series. It will be checked for plausibility and possibly used to improve the model at the next training.",
                duration=None
            )

            # Metadata from input sliders
            metadata = {
                "Latitude": input.lat(),
                "Longitude": input.lon(),
                "Turbine Type": input.unknown_turbine() if input.turbine_type() == 'unknown for model' else input.turbine_type(),
                "Hub Height": input.hub_height(),
                "Commissioning Year": input.commissioning_date_year(),
                "Commissioning Month": input.commissioning_date_month(),
                "Capacity (MW)": input.capacity(),
            }

            # Save folder and timestamp
            save_dir = os.path.join(root, "crowdsourced_data")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"time_series_{timestamp}.xlsx")

            # Combine metadata and time series data into a single DataFrame
            metadata_df = pd.DataFrame([metadata])  # Convert metadata to DataFrame

            # Save metadata and time series data to the same Excel file
            with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
                metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
                time_series_data.to_excel(writer, sheet_name="Time Series", index=False)


# Define absolute path to `www` folder
path_www = os.path.join(os.path.dirname(__file__), "..", "www")

# Start the app with the `www` directory as static content
app = App(app_ui, server, static_assets={"/": path_www})

if __name__ == "__main__":
    # Check if the variable `RENDER` is set to detect if the app is running on Render
    if os.getenv("RENDER") or os.getenv("WEBSITE_HOSTNAME"):  # for Render or Azure Server
        host = "0.0.0.0"  # For Render or other external deployments
    else:
        host = "127.0.0.1"  # For local development (localhost)

    app.run(host=host, port=8000)  # port binding: set the server port to 8000, because this is what Azure expects

# from ipywidgets import Dropdown
# from shinywidgets import output_widget, render_widget
# from shiny import App, ui

# app_ui = ui.page_fluid(output_widget("test"))

# def server(input, output, session):
#     @output
#     @render_widget
#     def test():
#         dropdown = Dropdown(
#             options=["A", "B", "C"],
#             value="A"
#         )
#         return dropdown

# app = App(app_ui, server)
# app.run()
# #app.run(host="0.0.0.0", port=8000)