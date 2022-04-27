#pragma once

#include <xaudio2.h>
#include <xaudio2fx.h>
#include <x3daudio.h>
#include <xapofx.h>
#pragma comment(lib,"xaudio2.lib")

#include <array>
#include <iostream>
#include <list>
#include <vector>
#include <string>
#include <math.h>       /* log */

#include <glm/glm.hpp>

#include "structures.hpp"
#include "VulkanDevice.h"

#define fourccRIFF 'FFIR'
#define fourccDATA 'atad'
#define fourccFMT ' tmf'
#define fourccWAVE 'EVAW'
#define fourccXWMA 'AMWX'
#define fourccDPDS 'sdpd'

HRESULT FindChunk(HANDLE hFile, DWORD fourcc, DWORD& dwChunkSize, DWORD& dwChunkDataPosition);
HRESULT ReadChunkData(HANDLE hFile, void* buffer, DWORD buffersize, DWORD bufferoffset);

namespace GAPT {
	// needs to be synchronized with the shaders
	const int ROWS = 2; // for each channel
	const int COLS = 32; // granularity for each channel
	// heuristic values for processing table results
	const int VOLUME_COEFFICIENT = 2.1;
	const int WET_COEFFICIENT = 0.85;

	class ResultsTable {
	protected:
		uint32_t sample_size;
	public:
		int data[ROWS][COLS];
		ResultsTable(uint32_t s) {
			sample_size = s;
			for( int i = 0; i < ROWS; i++ ) {
				for( int j = 0; j < COLS; j++ ) {
					data[i][j] = 0;
				}
			}
		}
		~ResultsTable() {
			//
		}

		float calculateVolume(int channel) {
			int front_length = 0.3 * COLS; // we want to truncate!
			float accumulator = 0;
			for( int i = 0; i < front_length; i++ ) {
				accumulator += data[channel][i];
			}
			return VOLUME_COEFFICIENT * accumulator / sample_size;
		}

		float calculateWetness(int channel) {
			int front_length = 0.3 * COLS; // we want to truncate!
			float front = 0;
			for( int i = 0; i < front_length; i++ ) {
				front += data[channel][i];
			}
			float back = 0.0;
			for( int i = front_length; i < COLS; i++ ) {
				back += data[channel][i];
			}
			return WET_COEFFICIENT * (back / front);
		}
	};

	class TemporalResults {
	protected:
		const int MAX_SIZE = 30;
		std::list<std::unique_ptr<ResultsTable>> lst;
	public:
		void AddNextData(std::unique_ptr<ResultsTable> next) {
			lst.emplace_front(std::move(next));
			if( MAX_SIZE < lst.size() ) {
				lst.resize(MAX_SIZE);
			}
		}

		float generateVolumeForChannel(int channel) {
			float coefficient = 1.0;
			float weighted_avg = 0.0; // accumulator
			for( auto it = lst.begin(); it != lst.end(); ++it ) {
				coefficient /= 2;
				weighted_avg += coefficient * it->get()->calculateWetness(channel);
			}
			weighted_avg /= (1 - coefficient);
			return weighted_avg;
		}

		float generateReverbForChannel(int channel) {
			float coefficient = 1.0;
			float weighted_avg = 0.0; // accumulator
			for( auto it = lst.begin(); it != lst.end(); ++it ) {
				coefficient /= 2;
				weighted_avg += coefficient * it->get()->calculateVolume(channel);
			}
			weighted_avg /= (1 - coefficient);
			return weighted_avg;
		}
	};

	class AudioData {
	protected:
		IXAudio2* engine;
		IXAudio2SourceVoice* pSourceVoice = nullptr;
		DWORD dwChunkSize;
		DWORD dwChunkPosition;
		WAVEFORMATEXTENSIBLE wfx = { 0 };
		XAUDIO2_BUFFER buffer = { 0 };
		IUnknown* reverb_effect;
		XAUDIO2_EFFECT_DESCRIPTOR descriptor;
		XAUDIO2_EFFECT_CHAIN chain;
		XAUDIO2FX_REVERB_PARAMETERS reverbParameters;
		glm::vec3 position;
		float volume;
		bool initialized;
		TemporalResults sim_results;
	public:
		AudioData(std::string& filepath, IXAudio2* pXAudio2) {
			initialized = false;
			engine = pXAudio2;
			HRESULT hr;

			// Open the file
			HANDLE hFile = CreateFileA(
				filepath.c_str(),
				GENERIC_READ,
				FILE_SHARE_READ,
				NULL,
				OPEN_EXISTING,
				0,
				NULL);

			if( INVALID_HANDLE_VALUE == hFile ) {
				std::cout << HRESULT_FROM_WIN32(GetLastError()) << "\n";
				return;
			}


			if( INVALID_SET_FILE_POINTER == SetFilePointer(hFile, 0, NULL, FILE_BEGIN) ) {
				std::cout << HRESULT_FROM_WIN32(GetLastError()) << "\n";
				return;
			}

			DWORD dwChunkSize;
			DWORD dwChunkPosition;
			//check the file type, should be fourccWAVE or 'XWMA'
			FindChunk(hFile, fourccRIFF, dwChunkSize, dwChunkPosition);
			DWORD filetype;
			ReadChunkData(hFile, &filetype, sizeof(DWORD), dwChunkPosition);
			if( filetype != fourccWAVE ) {
				return;
			}

			FindChunk(hFile, fourccFMT, dwChunkSize, dwChunkPosition);
			ReadChunkData(hFile, &wfx, dwChunkSize, dwChunkPosition);

			//fill out the audio data buffer with the contents of the fourccDATA chunk
			FindChunk(hFile, fourccDATA, dwChunkSize, dwChunkPosition);
			BYTE* pDataBuffer = new BYTE[dwChunkSize];
			ReadChunkData(hFile, pDataBuffer, dwChunkSize, dwChunkPosition);

			buffer.AudioBytes = dwChunkSize;  //size of the audio buffer in bytes
			buffer.pAudioData = pDataBuffer;  //buffer containing audio data
			buffer.Flags = XAUDIO2_END_OF_STREAM; // tell the source voice not to expect any data after this buffer

			if( FAILED(hr = engine->CreateSourceVoice(&pSourceVoice, (WAVEFORMATEX*)&wfx)) )
				return;

			/* OKAY, Now let's build the effect structures. */

			if( FAILED(hr = XAudio2CreateReverb(&reverb_effect)) ) {
				std::cout << "Failed to create reverb!\n";
				return;
			}
			descriptor.InitialState = true;
			descriptor.OutputChannels = 2;
			descriptor.pEffect = reverb_effect;

			chain.EffectCount = 1;
			chain.pEffectDescriptors = &descriptor;

			reverbParameters.DecayTime = XAUDIO2FX_REVERB_DEFAULT_DECAY_TIME;
			reverbParameters.Density = XAUDIO2FX_REVERB_DEFAULT_DENSITY;
			reverbParameters.PositionLeft = XAUDIO2FX_REVERB_DEFAULT_POSITION;
			reverbParameters.PositionRight = XAUDIO2FX_REVERB_DEFAULT_POSITION;
			reverbParameters.PositionMatrixLeft = XAUDIO2FX_REVERB_DEFAULT_POSITION_MATRIX;
			reverbParameters.PositionMatrixRight = XAUDIO2FX_REVERB_DEFAULT_POSITION_MATRIX;
			reverbParameters.EarlyDiffusion = XAUDIO2FX_REVERB_DEFAULT_EARLY_DIFFUSION;
			reverbParameters.LateDiffusion = XAUDIO2FX_REVERB_DEFAULT_LATE_DIFFUSION;
			reverbParameters.LowEQGain = XAUDIO2FX_REVERB_DEFAULT_LOW_EQ_GAIN;
			reverbParameters.LowEQCutoff = XAUDIO2FX_REVERB_DEFAULT_LOW_EQ_CUTOFF;
			reverbParameters.HighEQGain = XAUDIO2FX_REVERB_DEFAULT_HIGH_EQ_GAIN;
			reverbParameters.HighEQCutoff = XAUDIO2FX_REVERB_DEFAULT_HIGH_EQ_CUTOFF;
			reverbParameters.ReflectionsGain = XAUDIO2FX_REVERB_DEFAULT_REFLECTIONS_GAIN;
			reverbParameters.ReflectionsDelay = 50; // XAUDIO2FX_REVERB_DEFAULT_REFLECTIONS_DELAY; // 0 to 300 milliseconds
			reverbParameters.ReverbDelay = 50; //XAUDIO2FX_REVERB_DEFAULT_REVERB_DELAY; // 0 to 85 milliseconds
			reverbParameters.RearDelay = XAUDIO2FX_REVERB_DEFAULT_REAR_DELAY;
			reverbParameters.ReverbGain = XAUDIO2FX_REVERB_DEFAULT_REVERB_GAIN; // 
			reverbParameters.RoomFilterFreq = XAUDIO2FX_REVERB_DEFAULT_ROOM_FILTER_FREQ;
			reverbParameters.RoomFilterMain = XAUDIO2FX_REVERB_DEFAULT_ROOM_FILTER_MAIN;
			reverbParameters.RoomFilterHF = XAUDIO2FX_REVERB_DEFAULT_ROOM_FILTER_HF;
			reverbParameters.RoomSize = XAUDIO2FX_REVERB_DEFAULT_ROOM_SIZE;
			reverbParameters.WetDryMix = 50.0; //XAUDIO2FX_REVERB_DEFAULT_WET_DRY_MIX; // 0 to 100 scaling

			pSourceVoice->SetEffectChain(&chain);
			hr = pSourceVoice->SetEffectParameters(0, &reverbParameters, sizeof(reverbParameters));
			if( FAILED(hr) ) {
				std::cout << "Reverb Params failed!\n";
				return;
			}

			/* Oh hey w're done. */
			initialized = true;
		}

		~AudioData() {
			if( pSourceVoice ) delete pSourceVoice;
			if( buffer.pAudioData ) delete buffer.pAudioData;
			if( reverb_effect ) delete reverb_effect;
		}

		float* getVolume() { return &volume; }

		bool play() {
			if( !initialized )
				return false;

			HRESULT hr;
			pSourceVoice->EnableEffect(0);
			if( FAILED(hr = pSourceVoice->SubmitSourceBuffer(&buffer)) )
				return false;
			if( FAILED(hr = pSourceVoice->Start(0)) )
				return false;

			return true;
		}

		void addDataPoints(std::unique_ptr<ResultsTable> data) {
			sim_results.AddNextData(std::move(data));
			for( int i = 0; i < ROWS; i++ ) {
				auto vol = sim_results.generateVolumeForChannel(i);
				pSourceVoice->SetChannelVolumes(i, &vol);
				reverbParameters.WetDryMix = sim_results.generateReverbForChannel(i);
			}
		}

		void pause() {
			pSourceVoice->Stop();
		}

		void setPosition(glm::vec3& pos) {
			position = pos;
		}

		glm::vec3 getPosition() { return position; }
	};

	class PTAudio {
	protected:
		//device
		//global volume
		std::vector<AudioData*> sounds;
		IXAudio2* pXAudio2 = nullptr;
		IXAudio2MasteringVoice* pMasterVoice = nullptr;
		const int SAMPLE_SIZE = 10000;
		
		// compute pipeline related stuff
		AccelerationStructure* acceleration_structure;
		vks::VulkanDevice* vulkanDevice;
		struct {
			struct StorageBuffers {
				vks::Buffer input;
				vks::Buffer output;
			} storageBuffers;
			struct Semaphores {
				VkSemaphore ready{ 0L };
				VkSemaphore complete{ 0L };
			} semaphores;
			vks::Buffer uniformBuffer;
			VkQueue queue;
			VkCommandPool commandPool;
			std::array<VkCommandBuffer, 2> commandBuffers;
			VkFence waitFence;
			VkDescriptorSetLayout descriptorSetLayout;
			std::array<VkDescriptorSet, 2> descriptorSets;
			VkDescriptorPool descriptorPool;
			VkPipelineLayout pipelineLayout;
			VkPipeline pipeline;
			VkPipelineCache pipelineCache;
			VkDeviceMemory gpuResults;
			VkShaderModule computeShader;
			struct computeUBO {
				// these two values are the two factors of the total number of channels in a configuration
				int output_channels_rows, output_channels_cols;
				glm::vec3 source_location;
				glm::vec3 listener_position;
				glm::vec3 lookat;
				uint32_t bitmask = 0xFF;
				float listener_size;
			} ubo;
		} compute;

		bool initialized;
	public:
		PTAudio() {
			initialized = false;
			HRESULT hr;
			hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
			if( FAILED(hr) )
				return;

			pXAudio2 = nullptr;
			if( FAILED(hr = XAudio2Create(&pXAudio2, 0, XAUDIO2_DEFAULT_PROCESSOR)) )
				return;

			pMasterVoice = nullptr;
			if( FAILED(hr = pXAudio2->CreateMasteringVoice(&pMasterVoice)) )
				return;

			//auto dat = new AudioData(std::string("..\\data\\audio\\entertainer.wav"), pXAudio2);
			auto dat = new AudioData(std::string("..\\data\\audio\\skyrim.wav"), pXAudio2);
			sounds.push_back(dat);
			dat = new AudioData(std::string("..\\data\\audio\\snoring.wav"), pXAudio2);
			sounds.push_back(dat);
			dat = new AudioData(std::string("..\\data\\audio\\waterfall.wav"), pXAudio2);
			sounds.push_back(dat);

			// hard coded for now, but this isn't production code
			compute.ubo.output_channels_rows = 1;
			compute.ubo.output_channels_cols = 2;
			compute.ubo.listener_size = 0.15f;

			initialized = true;
		}

		~PTAudio() {
			if( pXAudio2 ) delete pXAudio2;
			for( int i = 0; i < sounds.size(); ++i ) {
				delete sounds[i];
			}
			//if( pMasterVoice ) delete pMasterVoice; // it looks like pMasterVoice is freed when pXAudio2 is freed ...?
			compute.storageBuffers.input.destroy();
			compute.storageBuffers.output.destroy();
			compute.uniformBuffer.destroy();
			vkDestroyPipelineLayout(vulkanDevice->logicalDevice, compute.pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(vulkanDevice->logicalDevice, compute.descriptorSetLayout, nullptr);
			vkDestroyPipeline(vulkanDevice->logicalDevice, compute.pipeline, nullptr);
			vkDestroySemaphore(vulkanDevice->logicalDevice, compute.semaphores.ready, nullptr);
			vkDestroySemaphore(vulkanDevice->logicalDevice, compute.semaphores.complete, nullptr);
			vkDestroyCommandPool(vulkanDevice->logicalDevice, compute.commandPool, nullptr);
		}

		void declareBitmask(uint32_t mask) {
			compute.ubo.bitmask = mask;
		}

		VkPipelineShaderStageCreateInfo loadShader(std::string fileName, VkShaderStageFlagBits stage) {
			VkPipelineShaderStageCreateInfo shaderStage = {};
			shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shaderStage.stage = stage;
			shaderStage.module = vks::tools::loadShader(fileName.c_str(), vulkanDevice->logicalDevice);
			shaderStage.pName = "main";
			assert(shaderStage.module != VK_NULL_HANDLE);
			compute.computeShader = shaderStage.module;
			return shaderStage;
		}

		AudioData* getSong(int id = 0) {
			return sounds[id];
		}

		void playSong(int sound_id) {
			if( !initialized ) return;
			if( sound_id < sounds.size() ) {
				sounds[sound_id]->play();
			}
		}

		void playAllSongs() {
			if( !initialized ) return;

			for( auto s : sounds ) {
				s->play();
			}
		}

		void setSoundLocation(int sound_id, glm::vec3& pos) {
			sounds[sound_id]->setPosition(pos);
		}

		void setListenerPosition(glm::vec3& pos, glm::vec3& lookat) {
			compute.ubo.listener_position = pos;
			compute.ubo.lookat = lookat;
		}

		void start() {
			playAllSongs(); // force the demo song
		}

		void mapUBOtoDevice(AudioData* data) {
			compute.ubo.source_location = data->getPosition();
			memcpy(compute.uniformBuffer.mapped, &compute.ubo, sizeof(compute.ubo));
		}

		void prepareComputeBuffers() {
			vulkanDevice->createBuffer(
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				&compute.uniformBuffer,
				sizeof(compute.ubo));

			// Create a compute capable device queue
			vkGetDeviceQueue(vulkanDevice->logicalDevice, vulkanDevice->queueFamilyIndices.compute, 0, &compute.queue);

			// Create compute pipeline
			std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
			};

			VkDescriptorSetLayoutCreateInfo descriptorLayout =
				vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);

			VK_CHECK_RESULT(vkCreateDescriptorSetLayout(vulkanDevice->logicalDevice, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

			VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
				vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayout, 1);

			// Push constants used to pass some parameters
			VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(uint32_t), 0);
			pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
			pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

			VK_CHECK_RESULT(vkCreatePipelineLayout(vulkanDevice->logicalDevice, &pipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout));

			VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(compute.descriptorPool, &compute.descriptorSetLayout, 1);

			// Create two descriptor sets with input and output buffers switched
			VK_CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->logicalDevice, &allocInfo, &compute.descriptorSets[0]));
			VK_CHECK_RESULT(vkAllocateDescriptorSets(vulkanDevice->logicalDevice, &allocInfo, &compute.descriptorSets[1]));

			std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
				vks::initializers::writeDescriptorSet(compute.descriptorSets[0], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, &compute.uniformBuffer.descriptor),
				vks::initializers::writeDescriptorSet(compute.descriptorSets[1], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, &compute.uniformBuffer.descriptor)
			};

			vkUpdateDescriptorSets(vulkanDevice->logicalDevice, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

			// Create pipeline
			VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
			computePipelineCreateInfo.stage = loadShader("shaders/computecloth/audio.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
			VK_CHECK_RESULT(vkCreateComputePipelines(vulkanDevice->logicalDevice, compute.pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline));

			// Separate command pool as queue family for compute may be different than graphics
			VkCommandPoolCreateInfo cmdPoolInfo = {};
			cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			cmdPoolInfo.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
			cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			VK_CHECK_RESULT(vkCreateCommandPool(vulkanDevice->logicalDevice, &cmdPoolInfo, nullptr, &compute.commandPool));

			// Create a command buffer for compute operations
			VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(compute.commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2);

			VK_CHECK_RESULT(vkAllocateCommandBuffers(vulkanDevice->logicalDevice, &cmdBufAllocateInfo, &compute.commandBuffers[0]));

			// Semaphores for graphics / compute synchronization
			VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
			VK_CHECK_RESULT(vkCreateSemaphore(vulkanDevice->logicalDevice, &semaphoreCreateInfo, nullptr, &compute.semaphores.ready));
			VK_CHECK_RESULT(vkCreateSemaphore(vulkanDevice->logicalDevice, &semaphoreCreateInfo, nullptr, &compute.semaphores.complete));

			// Build a single command buffer containing the compute dispatch commands
			buildComputeCommandBuffer();
		}

		void buildComputeCommandBuffer() {
			VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
			cmdBufInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

			uint32_t readSet = 0;
				VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffers[0], &cmdBufInfo));
				vkCmdBindPipeline(compute.commandBuffers[0], VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);

				// Dispatch the compute job
				vkCmdBindDescriptorSets(compute.commandBuffers[0], VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSets[readSet], 0, 0);
				uint32_t calculateNormals = 1;
				vkCmdPushConstants(compute.commandBuffers[0], compute.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &calculateNormals);

				vkCmdDispatch(compute.commandBuffers[0], 32, 1, 1);

				// release the storage buffers back to the graphics queue
				vkEndCommandBuffer(compute.commandBuffers[0]);
		}

		void runSimulation() {
			for( auto sound : sounds ) {
				compute.ubo.source_location = sound->getPosition();
				VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
				VkPipelineStageFlags computeWaitDstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
				computeSubmitInfo.waitSemaphoreCount = 1;
				computeSubmitInfo.pWaitSemaphores = &compute.semaphores.ready;
				computeSubmitInfo.pWaitDstStageMask = &computeWaitDstStageMask;
				computeSubmitInfo.signalSemaphoreCount = 1;
				computeSubmitInfo.pSignalSemaphores = &compute.semaphores.complete;
				computeSubmitInfo.commandBufferCount = 1;
				computeSubmitInfo.pCommandBuffers = &compute.commandBuffers[0];

				VK_CHECK_RESULT(vkQueueSubmit(compute.queue,
											  /*submitCount=*/1,
											  &computeSubmitInfo,
											  compute.waitFence));
				// Immediately wait for the result.
				// We're reusing buffers here.
				// Multiple buffers would probably run faster.
				vkWaitForFences(vulkanDevice->logicalDevice,
								/*fenceCount=*/1,
								&compute.waitFence,
								/*waitAll=*/VK_TRUE,
								/*timeout=*/UINT64_MAX);
				void* mapped;
				vkMapMemory(
					vulkanDevice->logicalDevice,
					compute.gpuResults,
					/*offset=*/0,
					VK_WHOLE_SIZE,
					/*flags=*/0,
					&mapped);
				VkMappedMemoryRange mappedRange = vks::initializers::mappedMemoryRange();
				mappedRange.memory = compute.gpuResults;
				mappedRange.offset = 0;
				mappedRange.size = VK_WHOLE_SIZE;
				const VkDeviceSize bufferSize = COLS * ROWS * sizeof(int);
				std::unique_ptr<ResultsTable> new_data(new ResultsTable(SAMPLE_SIZE));
				memcpy(new_data->data, mapped, bufferSize);
				sound->addDataPoints(std::move(new_data));
				vkUnmapMemory(vulkanDevice->logicalDevice, compute.gpuResults);
			}
		}

		// NOTE: We do NOT take ownership, only take a pointer to a shared accelerations structure resource.
		// We let the main engine deal with R/W data races and update data on it's terms.
		void UpdateEngineHooks(AccelerationStructure* as, vks::VulkanDevice* vkdev) {
			acceleration_structure = as;
			vulkanDevice = vkdev;
		}
	};
}