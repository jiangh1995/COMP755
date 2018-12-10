// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "example.hpp"          // Include short list of convenience functions for rendering
#include <chrono>
#include <atomic>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <imgui.h>
#include "imgui_impl_glfw.h"
#include <opencv2/opencv.hpp>   // Include OpenCV API

// Includes for time display
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>

using namespace std;

// Helper function for dispaying time conveniently
std::string pretty_time(std::chrono::nanoseconds duration);
// Helper function for rendering a seek bar
void draw_seek_bar(rs2::playback& playback, int* seek_pos, float2& location, float width);
float get_depth_scale(rs2::device dev);
rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);
rs2::config custom_config();

rs2::config custom_config(){
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 15);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 15);
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 15);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 15);
    return cfg;
}

struct filter_slider_ui
{
    std::string name;
    std::string label;
    std::string description;
    bool is_int;
    float value;
    rs2::option_range range;

    bool render(const float3& location, bool enabled);
    static bool is_all_integers(const rs2::option_range& range);
};

class filter_options
{
public:
    filter_options(const std::string name, rs2::processing_block& filter);
    filter_options(filter_options&& other);
    std::string filter_name;                                   //Friendly name of the filter
    rs2::processing_block& filter;                            //The filter in use
    std::map<rs2_option, filter_slider_ui> supported_options;  //maps from an option supported by the filter, to the corresponding slider
    std::atomic_bool is_enabled;                               //A boolean controlled by the user that determines whether to apply the filter or not
};

float get_depth_scale(rs2::device dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}

cv::Mat frame_to_mat(const rs2::frame& f)
{
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8)
    {
        return Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r = Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
        cvtColor(r, r, CV_RGB2BGR);
        return r;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16)
    {
        return Mat(Size(w, h), CV_16UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y8)
    {
        return Mat(Size(w, h), CV_8UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream align_to = RS2_STREAM_ANY;
    bool depth_stream_found = false;
    bool color_stream_found = false;
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream != RS2_STREAM_DEPTH)
        {
            if (!color_stream_found)         //Prefer color
                align_to = profile_stream;

            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
            }
        }
        else
        {
            depth_stream_found = true;
        }
    }

    if(!depth_stream_found)
        throw std::runtime_error("No Depth stream available");

    if (align_to == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");

    return align_to;
}

bool save_frame(rs2::frame frame, std::string filename){
    cv::Mat image = frame_to_mat(frame);
    cv::imwrite(filename, image);
    return true;
}
bool create_dir(std::string dirname){
    std::string create_base = std::string("mkdir -p ")+dirname;
    const int dir_err = system(create_base.c_str());
    return dir_err == -1;
}
std::string fixedLength(int value, int digits = 3) {
    unsigned int uvalue = value;
    if (value < 0) {
        uvalue = -uvalue;
    }
    std::string result;
    while (digits-- > 0) {
        result += ('0' + uvalue % 10);
        uvalue /= 10;
    }
    if (value < 0) {
        result += '-';
    }
    std::reverse(result.begin(), result.end());
    return result;
}



int main(int argc, char * argv[]) try
{
    if(argc<2){
        std::cout<<"Please input save dir"<<std::endl;
    }
    std::string save_file(argv[1]);
    create_dir(save_file);
    create_dir(save_file+string("/ir1/"));
    create_dir(save_file+string("/ir2/"));
    create_dir(save_file+string("/rgb/"));
    create_dir(save_file+string("/depth/"));

    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense Record and Playback Example");
    ImGui_ImplGlfw_Init(app, false);

    // Create booleans to control GUI (recorded - allow play button, recording - show 'recording to file' text)
    bool recorded = false;
    bool recording = false;

    // Declare a texture for the depth image on the GPU
    texture depth_image;
    texture ir1_image;
    texture ir2_image;
    texture rgb_image;

    // Declare frameset and frames which will hold the data from the camera
    rs2::frameset frames;
    rs2::frame depth;
    rs2::frame ir1;
    rs2::frame ir2;
    rs2::frame rgb;
    rs2::frame filtered_depth;
    rs2::frame raw_depth;

    rs2::spatial_filter spat_filter;    // Spatial    - edge-preserving spatial smoothing
    rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise

    // Initialize a vector that holds filters and their options
    std::vector<filter_options> filters;

    filters.emplace_back("Spatial", spat_filter);
    filters.emplace_back("Temporal", temp_filter);

    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    // Create a shared pointer to a pipeline
    auto pipe = std::make_shared<rs2::pipeline>();

    // Start streaming with default configuration
    pipe->start(custom_config());

    // Initialize a shared pointer to a device with the current device on the pipeline
    rs2::device device = pipe->get_active_profile().get_device();

    float depth_scale = get_depth_scale(device);

    rs2_stream align_to = find_stream_to_align(pipe->get_active_profile().get_streams());

    rs2::align align(align_to);

    // Create a variable to control the seek bar
    int seek_pos;

    bool is_recording = false;

    int count = 0;

    // While application is running
    while(app) {
        // Flags for displaying ImGui window
        static const int flags = ImGuiWindowFlags_NoCollapse
            | ImGuiWindowFlags_NoScrollbar
            | ImGuiWindowFlags_NoSavedSettings
            | ImGuiWindowFlags_NoTitleBar
            | ImGuiWindowFlags_NoResize
            | ImGuiWindowFlags_NoMove;

        ImGui_ImplGlfw_NewFrame(1);
        ImGui::SetNextWindowSize({ app.width(), app.height() });
        ImGui::Begin("app", nullptr, flags);

        // If the device is sreaming live and not from a file
        if (!device.as<rs2::playback>())
        {
            frames = pipe->wait_for_frames(); // wait for next set of frames from the camera
            frames = align.process(frames);
            rgb = frames.get_color_frame();
            raw_depth = frames.get_depth_frame();
            ir1 = frames.get_infrared_frame(1);
            ir2 = frames.get_infrared_frame(2);
            for (auto&& filter : filters)
            {
                if (filter.is_enabled)
                {
                    raw_depth = filter.filter.process(raw_depth);
                }
            }
            depth = color_map.process(raw_depth); // Find and colorize the depth data
            if (is_recording){
                string prefix = fixedLength(count, 6);
                save_frame(raw_depth, save_file + string("/depth/")+prefix+string(".png"));
                save_frame(rgb, save_file + string("/rgb/")+prefix+string(".png"));
                save_frame(ir1, save_file + string("/ir1/")+prefix+string(".png"));
                save_frame(ir2, save_file + string("/ir2/")+prefix+string(".png"));
                ++count;
            }
        }

        // Set options for the ImGui buttons
        ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, { 1, 1, 1, 1 });
        ImGui::PushStyleColor(ImGuiCol_Button, { 36 / 255.f, 44 / 255.f, 51 / 255.f, 1 });
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, { 40 / 255.f, 170 / 255.f, 90 / 255.f, 1 });
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, { 36 / 255.f, 44 / 255.f, 51 / 255.f, 1 });
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12);

        if (!device.as<rs2::playback>()) // Disable recording while device is playing
        {
            ImGui::SetCursorPos({ app.width() / 2 - 100, 3 * app.height() / 5 + 90});
            ImGui::Text("Click 'record' to start recording");
            ImGui::SetCursorPos({ app.width() / 2 - 100, 3 * app.height() / 5 + 110 });
            if (ImGui::Button("record", { 50, 50 }))
            {
                // If it is the start of a new recording (device is not a recorder yet)
                if (!device.as<rs2::recorder>())
                {
                    is_recording = true;
                    /*pipe->stop(); // Stop the pipeline with the default configuration
                    pipe = std::make_shared<rs2::pipeline>();
                    rs2::config cfg = custom_config(); // Declare a new configuration
                    cfg.enable_record_to_file(save_file);
                    pipe->start(cfg); //File will be opened at this point
                    device = pipe->get_active_profile().get_device();*/

                }
                else
                { // If the recording is resumed after a pause, there's no need to reset the shared pointer
                    device.as<rs2::recorder>().resume(); // rs2::recorder allows access to 'resume' function
                }
                recording = true;
            }

            /*
            When pausing, device still holds the file.
            */
            if (device.as<rs2::recorder>())
            {
                if (recording)
                {
                    ImGui::SetCursorPos({ app.width() / 2 - 100, 3 * app.height() / 5 + 60 });
                    ImGui::TextColored({ 255 / 255.f, 64 / 255.f, 54 / 255.f, 1 }, (std::string("Recording to file ")+save_file).c_str());
                }

                // Pause the playback if button is clicked
                ImGui::SetCursorPos({ app.width() / 2, 3 * app.height() / 5 + 110 });
                if (ImGui::Button("pause\nrecord", { 50, 50 }))
                {
                    device.as<rs2::recorder>().pause();
                    recording = false;
                }

                ImGui::SetCursorPos({ app.width() / 2 + 100, 3 * app.height() / 5 + 110 });
                if (ImGui::Button(" stop\nrecord", { 50, 50 }))
                {
                    pipe->stop(); // Stop the pipeline that holds the file and the recorder
                    pipe = std::make_shared<rs2::pipeline>(); //Reset the shared pointer with a new pipeline
                    pipe->start(custom_config()); // Resume streaming with default configuration
                    device = pipe->get_active_profile().get_device();
                    recorded = true; // Now we can run the file
                    recording = false;
                }
            }
        }

        // After a recording is done, we can play it
        if (recorded) {
            ImGui::SetCursorPos({ app.width() / 2 - 100, 4 * app.height() / 5 + 30 });
            ImGui::Text("Click 'play' to start playing");
            ImGui::SetCursorPos({ app.width() / 2 - 100, 4 * app.height() / 5 + 50});
            if (ImGui::Button("play", { 50, 50 }))
            {
                if (!device.as<rs2::playback>())
                {
                    pipe->stop(); // Stop streaming with default configuration
                    pipe = std::make_shared<rs2::pipeline>();
                    rs2::config cfg;
                    cfg.enable_device_from_file(save_file);
                    pipe->start(cfg); //File will be opened in read mode at this point
                    device = pipe->get_active_profile().get_device();
                }
                else
                {
                    device.as<rs2::playback>().resume();
                }
            }
        }

        // If device is playing a recording, we allow pause and stop
        if (device.as<rs2::playback>())
        {
            rs2::playback playback = device.as<rs2::playback>();
            if (pipe->poll_for_frames(&frames)) // Check if new frames are ready
            {
                depth = color_map.process(frames.get_depth_frame()); // Find and colorize the depth data for rendering
                rgb = frames.get_color_frame();
                //depth = frames.get_depth_frame();
                ir1 = frames.get_infrared_frame(1);
                ir2 = frames.get_infrared_frame(2);
            }

            // Render a seek bar for the player
            float2 location = { app.width() / 4, 4 * app.height() / 5 + 110 };
            draw_seek_bar(playback , &seek_pos, location, app.width() / 2);

            ImGui::SetCursorPos({ app.width() / 2, 4 * app.height() / 5 + 50 });
            if (ImGui::Button(" pause\nplaying", { 50, 50 }))
            {
                playback.pause();
            }

            ImGui::SetCursorPos({ app.width() / 2 + 100, 4 * app.height() / 5 + 50 });
            if (ImGui::Button("  stop\nplaying", { 50, 50 }))
            {
                pipe->stop();
                pipe = std::make_shared<rs2::pipeline>();
                pipe->start(custom_config());
                device = pipe->get_active_profile().get_device();
            }
        }

        ImGui::PopStyleColor(4);
        ImGui::PopStyleVar();

        ImGui::End();
        ImGui::Render();

        // Render depth frames from the default configuration, the recorder or the playback
        depth_image.render(depth, { 0, 0,  app.width() / 3, app.height() / 3});
        rgb_image.render(rgb, { app.width()/2, 0,  app.width() / 3, app.height() / 3});
        ir1_image.render(ir1, { 0, app.height()/3,  app.width() / 3, app.height() / 3});
        ir2_image.render(ir2, { app.width()/2, app.height()/3,  app.width() / 3, app.height() / 3});

    }
    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cout << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}


std::string pretty_time(std::chrono::nanoseconds duration)
{
    using namespace std::chrono;
    auto hhh = duration_cast<hours>(duration);
    duration -= hhh;
    auto mm = duration_cast<minutes>(duration);
    duration -= mm;
    auto ss = duration_cast<seconds>(duration);
    duration -= ss;
    auto ms = duration_cast<milliseconds>(duration);

    std::ostringstream stream;
    stream << std::setfill('0') << std::setw(hhh.count() >= 10 ? 2 : 1) << hhh.count() << ':' <<
        std::setfill('0') << std::setw(2) << mm.count() << ':' <<
        std::setfill('0') << std::setw(2) << ss.count();
    return stream.str();
}


void draw_seek_bar(rs2::playback& playback, int* seek_pos, float2& location, float width)
{
    int64_t playback_total_duration = playback.get_duration().count();
    auto progress = playback.get_position();
    double part = (1.0 * progress) / playback_total_duration;
    *seek_pos = static_cast<int>(std::max(0.0, std::min(part, 1.0)) * 100);
    auto playback_status = playback.current_status();
    ImGui::PushItemWidth(width);
    ImGui::SetCursorPos({ location.x, location.y });
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12);
    if (ImGui::SliderInt("##seek bar", seek_pos, 0, 100, "", true))
    {
        //Seek was dragged
        if (playback_status != RS2_PLAYBACK_STATUS_STOPPED) //Ignore seek when playback is stopped
        {
            auto duration_db = std::chrono::duration_cast<std::chrono::duration<double, std::nano>>(playback.get_duration());
            auto single_percent = duration_db.count() / 100;
            auto seek_time = std::chrono::duration<double, std::nano>((*seek_pos) * single_percent);
            playback.seek(std::chrono::duration_cast<std::chrono::nanoseconds>(seek_time));
        }
    }
    std::string time_elapsed = pretty_time(std::chrono::nanoseconds(progress));
    ImGui::SetCursorPos({ location.x + width + 10, location.y });
    ImGui::Text("%s", time_elapsed.c_str());
    ImGui::PopStyleVar();
    ImGui::PopItemWidth();
}
filter_options::filter_options(const std::string name, rs2::processing_block& filter) :
    filter_name(name),
    filter(filter),
    is_enabled(true)
{
    const std::array<rs2_option, 3> possible_filter_options = {
        RS2_OPTION_FILTER_MAGNITUDE,
        RS2_OPTION_FILTER_SMOOTH_ALPHA,
        RS2_OPTION_FILTER_SMOOTH_DELTA
    };
    //Go over each filter option and create a slider for it
    for (rs2_option opt : possible_filter_options)
    {
        if (filter.supports(opt))
        {
            rs2::option_range range = filter.get_option_range(opt);
            supported_options[opt].range = range;
            supported_options[opt].value = range.def;
            supported_options[opt].is_int = filter_slider_ui::is_all_integers(range);
            supported_options[opt].description = filter.get_option_description(opt);
            std::string opt_name = rs2_option_to_string(opt);
            supported_options[opt].name = name + "_" + opt_name;
            std::string prefix = "Filter ";
            supported_options[opt].label = name + " " + opt_name.substr(prefix.length());
        }
    }
}

filter_options::filter_options(filter_options&& other) :
    filter_name(std::move(other.filter_name)),
    filter(other.filter),
    supported_options(std::move(other.supported_options)),
    is_enabled(other.is_enabled.load())
{
}

bool filter_slider_ui::render(const float3& location, bool enabled)
{
    bool value_changed = false;
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, { 40 / 255.f, 170 / 255.f, 90 / 255.f, 1 });
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, { 20 / 255.f, 150 / 255.f, 70 / 255.f, 1 });
    ImGui::GetStyle().GrabRounding = 12;
    if (!enabled)
    {
        ImGui::PushStyleColor(ImGuiCol_SliderGrab, { 0,0,0,0 });
        ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, { 0,0,0,0 });
        ImGui::PushStyleColor(ImGuiCol_Text, { 0.6f, 0.6f, 0.6f, 1 });
    }

    ImGui::PushItemWidth(location.z);
    ImGui::SetCursorPos({ location.x, location.y + 3 });
    ImGui::TextUnformatted(label.c_str());
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", description.c_str());

    ImGui::SetCursorPos({ location.x + 170, location.y });

    if (is_int)
    {
        int value_as_int = static_cast<int>(value);
        value_changed = ImGui::SliderInt(("##" + name).c_str(), &value_as_int, static_cast<int>(range.min), static_cast<int>(range.max), "%.0f");
        value = static_cast<float>(value_as_int);
    }
    else
    {
        value_changed = ImGui::SliderFloat(("##" + name).c_str(), &value, range.min, range.max, "%.3f", 1.0f);
    }

    ImGui::PopItemWidth();

    if (!enabled)
    {
        ImGui::PopStyleColor(3);
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
    return value_changed;
}

/**
  Helper function for deciding on int ot float slider
*/
bool filter_slider_ui::is_all_integers(const rs2::option_range& range)
{
    const auto is_integer = [](float f)
    {
        return (fabs(fmod(f, 1)) < std::numeric_limits<float>::min());
    };

    return is_integer(range.min) && is_integer(range.max) &&
        is_integer(range.def) && is_integer(range.step);
}