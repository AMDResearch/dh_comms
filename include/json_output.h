#pragma once

#include <json.h>
#include <string>
#include <chrono>
#include <fstream>
#include <memory>
#include <iostream>
#include <mutex>

namespace dh_comms {

using json = nlohmann::json;

class JsonOutputManager {
public:
    static JsonOutputManager& getInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = new JsonOutputManager();
            std::cout << "[JsonOutputManager] Created new instance at " << (void*)instance_ << std::endl;
        }
        return *instance_;
    }

    // Add cleanup method to be called when we're truly done with all analysis
    static void cleanup() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (instance_) {
            std::cout << "[JsonOutputManager] Cleaning up instance at " << (void*)instance_ << std::endl;
            delete instance_;
            instance_ = nullptr;
        }
    }

    // Prevent copying
    JsonOutputManager(const JsonOutputManager&) = delete;
    JsonOutputManager& operator=(const JsonOutputManager&) = delete;

    void initializeKernelAnalysis(const std::string& kernel_name, uint64_t dispatch_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "[JsonOutputManager:" << (void*)this << "] Initializing kernel analysis for " << kernel_name 
                  << " (dispatch_id: " << dispatch_id << ")" << std::endl;
        
        // Reset the analysis if it was cleared
        if (current_analysis_.is_null()) {
            current_analysis_ = json::object();
        }
        
        if (!current_analysis_.contains("kernel_analyses")) {
            current_analysis_["kernel_analyses"] = json::array();
        }

        json kernel_analysis;
        kernel_analysis["kernel_info"]["name"] = kernel_name;
        kernel_analysis["kernel_info"]["dispatch_id"] = dispatch_id;
        kernel_analysis["cache_analysis"]["accesses"] = json::array();
        kernel_analysis["bank_conflicts"]["accesses"] = json::array();
        
        current_analysis_["kernel_analyses"].push_back(kernel_analysis);
        std::cout << "[JsonOutputManager:" << (void*)this << "] Kernel analysis initialized, size: " 
                  << current_analysis_["kernel_analyses"].size() << std::endl;
    }

    void setMetadata(const std::string& gpu_arch, uint32_t cache_line_size, size_t kernels_found) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "[JsonOutputManager:" << (void*)this << "] Setting metadata - GPU: " << gpu_arch 
                  << ", Cache line size: " << cache_line_size 
                  << ", Kernels found: " << kernels_found << std::endl;
                  
        // Get current time and format it
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now = *std::localtime(&time_t_now);
        char timestamp[32];
        std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &tm_now);
        
        current_analysis_["metadata"] = {
            {"timestamp", std::string(timestamp)},
            {"version", "1.0"},
            {"gpu_info", {
                {"architecture", gpu_arch},
                {"cache_line_size", cache_line_size}
            }},
            {"kernels_found", kernels_found}
        };
        
        std::cout << "[JsonOutputManager:" << (void*)this << "] Metadata set successfully" << std::endl;
    }

    void updateKernelsFound(size_t kernels_found) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "[JsonOutputManager:" << (void*)this << "] Updating kernels found to: " << kernels_found << std::endl;
        if (current_analysis_.contains("metadata")) {
            current_analysis_["metadata"]["kernels_found"] = kernels_found;
        }
    }

    void addCacheAnalysis(const std::string& file, uint32_t line, uint32_t column,
                         const std::string& code_context, const std::string& access_type,
                         uint16_t ir_bytes, uint16_t isa_bytes, const std::string& isa_instruction,
                         size_t execution_count, size_t cache_lines_needed, size_t cache_lines_used) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "[JsonOutputManager:" << (void*)this << "] Adding cache analysis for " << file << ":" << line << ":" << column << std::endl
                  << "  Access type: " << access_type << ", IR bytes: " << ir_bytes << ", ISA bytes: " << isa_bytes << std::endl
                  << "  Cache lines needed: " << cache_lines_needed << ", used: " << cache_lines_used << std::endl;
                  
        if (current_analysis_["kernel_analyses"].empty()) {
            std::cout << "[JsonOutputManager:" << (void*)this << "] WARNING: No kernel analyses found when adding cache analysis" << std::endl;
            return;
        }

        json access;
        access["source_location"] = {
            {"file", file},
            {"line", line},
            {"column", column}
        };
        access["code_context"] = code_context;
        access["access_info"] = {
            {"type", access_type},
            {"ir_bytes", ir_bytes},
            {"isa_bytes", isa_bytes},
            {"isa_instruction", isa_instruction},
            {"execution_count", execution_count},
            {"cache_lines", {
                {"needed", cache_lines_needed},
                {"used", cache_lines_used}
            }}
        };

        auto& current_kernel = current_analysis_["kernel_analyses"].back();
        current_kernel["cache_analysis"]["accesses"].push_back(access);
        
        std::cout << "[JsonOutputManager:" << (void*)this << "] Added cache analysis, current size: " 
                  << current_analysis_["kernel_analyses"].size() << std::endl;
    }

    void addBankConflict(const std::string& file, uint32_t line, uint32_t column,
                        const std::string& code_context, const std::string& access_type,
                        uint16_t ir_bytes, size_t execution_count, size_t total_conflicts) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "[JsonOutputManager:" << (void*)this << "] Adding bank conflict for " << file << ":" << line << ":" << column << std::endl
                  << "  Access type: " << access_type << ", IR bytes: " << ir_bytes << std::endl
                  << "  Execution count: " << execution_count << ", Total conflicts: " << total_conflicts << std::endl;
                  
        if (current_analysis_["kernel_analyses"].empty()) {
            std::cout << "[JsonOutputManager:" << (void*)this << "] WARNING: No kernel analyses found when adding bank conflict" << std::endl;
            return;
        }

        json access;
        access["source_location"] = {
            {"file", file},
            {"line", line},
            {"column", column}
        };
        access["code_context"] = code_context;
        access["access_info"] = {
            {"type", access_type},
            {"ir_bytes", ir_bytes},
            {"execution_count", execution_count},
            {"total_conflicts", total_conflicts}
        };

        auto& current_kernel = current_analysis_["kernel_analyses"].back();
        current_kernel["bank_conflicts"]["accesses"].push_back(access);
        
        std::cout << "[JsonOutputManager:" << (void*)this << "] Added bank conflict, current size: " 
                  << current_analysis_["kernel_analyses"].size() << std::endl;
    }

    void setExecutionTimes(uint64_t start_ns, uint64_t end_ns, uint64_t complete_ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_analysis_["kernel_analyses"].empty()) {
            std::cout << "[JsonOutputManager:" << (void*)this << "] WARNING: No kernel analyses found when setting execution times" << std::endl;
            return;
        }

        auto& current_kernel = current_analysis_["kernel_analyses"].back();
        current_kernel["kernel_info"]["execution_time"] = {
            {"start_ns", start_ns},
            {"end_ns", end_ns},
            {"complete_ns", complete_ns}
        };
    }

    void setProcessingStats(size_t bytes_processed, double processing_time_seconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_analysis_["kernel_analyses"].empty()) {
            std::cout << "[JsonOutputManager:" << (void*)this << "] WARNING: No kernel analyses found when setting processing stats" << std::endl;
            return;
        }

        auto& current_kernel = current_analysis_["kernel_analyses"].back();
        current_kernel["kernel_info"]["bytes_processed"] = bytes_processed;
        current_kernel["kernel_info"]["processing_time_seconds"] = processing_time_seconds;
        current_kernel["kernel_info"]["throughput_mib_per_sec"] = 
            (bytes_processed / processing_time_seconds) / 1.0e6;
    }

    void writeToFile(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "[JsonOutputManager:" << (void*)this << "] Writing analysis to file: " << filename << std::endl;
        std::ofstream out(filename);
        out << current_analysis_.dump(2);
        std::cout << "[JsonOutputManager:" << (void*)this << "] File write complete" << std::endl;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "[JsonOutputManager:" << (void*)this << "] Clearing all data" << std::endl;
        current_analysis_ = json::object();  // Reset to empty object instead of clearing
    }

    // Debug method to get current analysis size
    size_t getCurrentAnalysisSize() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!current_analysis_.contains("kernel_analyses")) {
            return 0;
        }
        return current_analysis_["kernel_analyses"].size();
    }

    // Debug method to dump current state
    void dumpCurrentState() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << "\n[JsonOutputManager:" << (void*)this << "] Current State:" << std::endl;
        std::cout << current_analysis_.dump(2) << std::endl;
    }

private:
    JsonOutputManager() {
        std::cout << "[JsonOutputManager] Constructor called for instance " << (void*)this << std::endl;
        current_analysis_ = json::object();  // Initialize as empty object instead of null
    }
    
    ~JsonOutputManager() {
        std::cout << "[JsonOutputManager] Destructor called for instance " << (void*)this << std::endl;
    }
    
    mutable std::mutex mutex_;
    json current_analysis_;

    // Static members for singleton - defined in json_output.cpp
    static JsonOutputManager* instance_;
    static std::mutex instance_mutex_;
}; 

} // namespace dh_comms 