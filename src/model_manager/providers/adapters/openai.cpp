#include "flock/model_manager/providers/adapters/openai.hpp"
#include "flock/model_manager/model.hpp"
#include "flock/model_manager/providers/handlers/url_handler.hpp"
#include <fmt/format.h>
#include <algorithm>
#include <cctype>

namespace flock {

void OpenAIProvider::AddCompletionRequest(const std::string& prompt, const int num_output_tuples, OutputType output_type, const nlohmann::json& media_data) {
    auto message_content = nlohmann::json::array();

    message_content.push_back({{"type", "text"}, {"text", prompt}});

    // Process image columns
    if (media_data.contains("image") && !media_data["image"].empty() && media_data["image"].is_array()) {
        for (const auto& column: media_data["image"]) {
            const auto detail = column.contains("detail") ? column["detail"].get<std::string>() : "low";
            auto image_type = column.contains("type") ? column["type"].get<std::string>() : "image";
            message_content.push_back(
                    {{"type", "text"},
                     {"text", "ATTACHMENT COLUMN"}});
            auto row_index = 1u;
            for (const auto& image: column["data"]) {
                // Skip null values
                if (image.is_null()) {
                    row_index++;
                    continue;
                }
                auto image_url = std::string();
                std::string image_str;
                if (image.is_string()) {
                    image_str = image.get<std::string>();
                } else {
                    image_str = image.dump();
                }
                if (image_str.empty()) {
                    row_index++;
                    continue;
                }
                message_content.push_back(
                        {{"type", "text"}, {"text", "ROW " + std::to_string(row_index) + " :"}});

                auto mime_type = image_type.rfind("image/", 0) == 0 ? image_type : std::string();
                if (mime_type.empty()) {
                    auto extension = URLHandler::ExtractFileExtension(image_str);
                    std::transform(extension.begin(), extension.end(), extension.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    if (extension == ".jpg" || extension == ".jpeg") mime_type = "image/jpeg";
                    else if (extension == ".webp") mime_type = "image/webp";
                    else if (extension == ".gif") mime_type = "image/gif";
                    else mime_type = "image/png";
                }

                // Handle file path or URL
                if (URLHandler::IsUrl(image_str) || image_str.rfind("data:", 0) == 0) {
                    // URL - send directly to API
                    image_url = image_str;
                } else {
                    // File path - read and convert to base64
                    auto base64_result = URLHandler::ResolveFileToBase64(image_str);
                    image_url = duckdb_fmt::format("data:{};base64,{}", mime_type, base64_result.base64_content);
                }

                message_content.push_back(
                        {{"type", "image_url"},
                         {"image_url", {{"url", image_url}, {"detail", detail}}}});
                row_index++;
            }
        }
    }

    nlohmann::json request_payload = {{"model", model_details_.model},
                                      {"messages", {{{"role", "user"}, {"content", message_content}}}}};

    if (!model_details_.model_parameters.empty()) {
        request_payload.update(model_details_.model_parameters);
    }

    if (model_details_.model_parameters.contains("response_format")) {
        auto schema = model_details_.model_parameters["response_format"]["json_schema"]["schema"];
        auto strict = model_details_.model_parameters["response_format"]["strict"];
        request_payload["response_format"] = {
                {"type", "json_schema"},
                {"json_schema",
                 {{"name", "flock_response"},
                  {"strict", strict},
                  {"schema", {{"type", "object"}, {"properties", {{"items", {{"type", "array"}, {"minItems", num_output_tuples}, {"maxItems", num_output_tuples}, {"items", schema}}}}}, {"required", {"items"}}, {"additionalProperties", false}}}}}};
    } else {
        request_payload["response_format"] = {
                {"type", "json_schema"},
                {"json_schema",
                 {{"name", "flock_response"},
                  {"strict", false},
                  {"schema", {{"type", "object"}, {"properties", {{"items", {{"type", "array"}, {"minItems", num_output_tuples}, {"maxItems", num_output_tuples}, {"items", {{"type", GetOutputTypeString(output_type)}}}}}}}}}}}};
        // Require "items" + forbid extra keys, matching the model_parameters branch
        // above and the semantic operator's schema. WITHOUT this, vLLM guided
        // decoding lets the model satisfy the schema with "{}" -> no items -> padded
        // null -> defaulted to "true", so EVERY row passes (the scalar all-true bug).
        auto& sjs = request_payload["response_format"]["json_schema"]["schema"];
        sjs["required"] = nlohmann::json::array({"items"});
        sjs["additionalProperties"] = false;
    }

    model_handler_->AddRequest(request_payload);
}

void OpenAIProvider::AddEmbeddingRequest(const std::vector<std::string>& inputs) {
    nlohmann::json request_payload = {
            {"model", model_details_.model},
            {"input", inputs},
    };

    model_handler_->AddRequest(request_payload, IModelProviderHandler::RequestType::Embedding);
}

void OpenAIProvider::AddTranscriptionRequest(const nlohmann::json& audio_files) {
    for (const auto& audio_file: audio_files) {
        // Skip null values
        if (audio_file.is_null()) {
            continue;
        }
        std::string audio_file_str;
        if (audio_file.is_string()) {
            audio_file_str = audio_file.get<std::string>();
        } else {
            audio_file_str = audio_file.dump();
        }

        // Handle file download and validation
        auto file_result = URLHandler::ResolveFilePath(audio_file_str);

        nlohmann::json transcription_request = {
                {"file_path", file_result.file_path},
                {"model", model_details_.model},
                {"is_temp_file", file_result.is_temp_file}};
        model_handler_->AddRequest(transcription_request, IModelProviderHandler::RequestType::Transcription);
    }
}

}// namespace flock
