%% Part (c-d)
clear; close all; clc;

%% ========================= 1) LOAD + TRIM AUDIO =========================
fprintf('--- Loading audio ---\n');

audioFile  = "Ryan Gosling - City Of Stars (320).mp3";
maxSeconds = 4;

[srcSig, fs] = loadMonoAudio(audioFile, maxSeconds);
t = (0:numel(srcSig)-1)'/fs;

%% ===================== 2) ARRAY + SOURCE TRAJECTORY SETUP =====================
fprintf('--- Building geometry ---\n');

cSound      = 343;      % m/s
numMics     = 15;       % N
micSpacing  = 0.18;     % d [m]
L           = 2.0;      % ellipse parameter [m]
omega0      = pi/2;     % rad/s
refMicIdx   = ceil(numMics/2);   % 8 for N=15

micPosXY    = buildLinearArray(numMics, micSpacing);
srcPosXY    = ellipticalSourceTrajectory(t, L, omega0);

%% ================== 3) PROPAGATION: DELAY + ATTENUATION ==================
fprintf('--- Simulating propagation (delay + attenuation) ---\n');

r0 = 1;
[micClean, attMat] = simulatePropagation(srcSig, t, fs, srcPosXY, micPosXY, cSound, r0); % [Ns x N]

%% ===================== 4) ADD CORRELATED NEIGHBOR NOISE =====================
fprintf('--- Adding correlated neighbor noise ---\n');

rhoNeighbor = 0.2;
targetSNR_dB = 20;
micNoisy = addNeighborCorrelatedNoise(micClean, targetSNR_dB, rhoNeighbor, 0);

%% ============== 5) WINDOWED DELAY ESTIMATION (vs center mic) ===============
fprintf('\n--- Estimating time-varying delays (windowed xcorr) ---\n');

winSec      = 0.15;      % 150 ms
overlapFrac = 0.50;      % 50% overlap

[delayHat_sec, delayTrue_sec, winTime_sec, xcorrPeak] = ...
    estimateDelaysWindowed(micNoisy, t, fs, srcPosXY, micPosXY, cSound, refMicIdx, winSec, overlapFrac);

% Smooth delay estimates across blocks (toolbox-free)
delayHat_sec = smoothBlockDelays(delayHat_sec, refMicIdx, 7);

% RMS delay error per mic (ms)
rmsDelayErr_ms = computeRmsDelayError(delayHat_sec, delayTrue_sec);

%% ========================= 6) PLOTS: DELAY QUALITY =========================

figure('Name','Delay Tracks Check (Mic 1 & Mic 15 vs Reference)');

subplot(2,1,1);
plot(winTime_sec, 1000*delayTrue_sec(1,:), 'k-', 'LineWidth', 1.6); hold on;
plot(winTime_sec, 1000*delayHat_sec(1,:),  'r--','LineWidth', 1.6);
grid on; xlabel('Time (s)'); ylabel('Delay (ms)');
title(sprintf('Delay Tracks vs Ref Mic %d (Mic 1)', refMicIdx));
legend('True','Estimated','Location','best');

subplot(2,1,2);
plot(winTime_sec, 1000*delayTrue_sec(end,:), 'b-', 'LineWidth', 1.6); hold on;
plot(winTime_sec, 1000*delayHat_sec(end,:),  'm--','LineWidth', 1.6);
grid on; xlabel('Time (s)'); ylabel('Delay (ms)');
title(sprintf('Delay Tracks vs Ref Mic %d (Mic %d)', refMicIdx, numMics));
legend('True','Estimated','Location','best');

%% ================== 7) ALIGN ALL CHANNELS USING DELAY ESTIMATES ==================
fprintf('\n--- Aligning all microphone signals ---\n');

delayPerSample_sec = interpolateDelaysToSamples(delayHat_sec, winTime_sec, t, refMicIdx);

alignedNoisy = applyTimeAlignment(micNoisy, fs, delayPerSample_sec);   % [Ns x N]
alignedClean = applyTimeAlignment(micClean, fs, delayPerSample_sec);

% Align attenuation trajectories the SAME way (so equalization is consistent)
attAligned   = applyTimeAlignment(attMat, fs, delayPerSample_sec);

% --- Gain correction (undo attenuation) ---
% Equalizing by 1/att can amplify noise when source is far; so we use a floor to avoid blow-ups.
attFloor = prctile(attAligned(:), 10);      % 10th percentile floor
attUse   = max(attAligned, attFloor);

epsA = 1e-6;
alignedNoisy_eq = alignedNoisy ./ (attUse + epsA);
alignedClean_eq = alignedClean ./ (attUse + epsA);

%% ===================== 8) PART (c): SIMPLE AVERAGE =====================
yAvg       = mean(alignedNoisy_eq, 2);
yAvg_clean = mean(alignedClean_eq, 2);
yAvg_noise = yAvg - yAvg_clean;

%% ===================== VALID REGION + INPUT SNR  =====================
validIdx = (floor(0.5*fs)+1) : (numel(srcSig)-floor(0.2*fs));

refNoise = micNoisy(:,refMicIdx) - micClean(:,refMicIdx);
snrIn_ref_dB = 10*log10( var(micClean(validIdx,refMicIdx)) / max(var(refNoise(validIdx)),1e-12) );

%% ===================== 9) PART (d-1): THEORETICAL BLUE (MODEL) =====================
fprintf('\n--- Part (d-1): Theoretical BLUE (model covariance) ---\n');

Rn_model = neighborCovariance(numMics, 1.0, rhoNeighbor);
one = ones(numMics,1);
wOpt_model = (Rn_model \ one) / (one' * (Rn_model \ one));   % sum(w)=1

yOpt_model       = alignedNoisy_eq * wOpt_model;
yOpt_model_clean = alignedClean_eq * wOpt_model;
yOpt_model_noise = yOpt_model - yOpt_model_clean;

snrOut_model_dB = 10*log10( var(yOpt_model_clean(validIdx)) / max(var(yOpt_model_noise(validIdx)),1e-12) );
fprintf('Output SNR (Model BLUE): %.2f dB | Gain vs ref: %.2f dB\n', snrOut_model_dB, snrOut_model_dB - snrIn_ref_dB);

%% ===================== 10) PART (d-2): EMPIRICAL BLUE (POST-PROCESS NOISE) =====================
fprintf('\n--- Part (d-2): Optimal weighting using empirical noise covariance ---\n');

nEq = alignedNoisy_eq - alignedClean_eq;          % noise at combiner input
nEq_valid = nEq(validIdx, :);

Rn_emp = (nEq_valid' * nEq_valid) / size(nEq_valid,1);       % N x N

% Diagonal loading for stability
dl = 1e-6 * trace(Rn_emp)/numMics;
Rn_emp = Rn_emp + dl*eye(numMics);

wOpt = (Rn_emp \ one) / (one' * (Rn_emp \ one));             % sum(w)=1
wAvg = one/numMics;

% Guarantee-check 
nv_avg = wAvg' * Rn_emp * wAvg;
nv_opt = wOpt' * Rn_emp * wOpt;
fprintf('Empirical noise var (avg): %.6f\n', nv_avg);
fprintf('Empirical noise var (opt): %.6f\n', nv_opt);
fprintf('Empirical improvement:    %.4f dB\n', 10*log10(nv_avg/nv_opt));
fprintf('Sum of empirical optimal weights: %.6f\n', sum(wOpt));

yOpt       = alignedNoisy_eq * wOpt;
yOpt_clean = alignedClean_eq * wOpt;
yOpt_noise = yOpt - yOpt_clean;

snrOut_opt_dB = 10*log10( var(yOpt_clean(validIdx)) / max(var(yOpt_noise(validIdx)),1e-12) );
fprintf('Output SNR (Optimal, empirical): %.2f dB | Gain vs ref: %.2f dB\n', snrOut_opt_dB, snrOut_opt_dB - snrIn_ref_dB);

%% ===================== 11) GLOBAL SNR + MSE SUMMARY =====================
snrOut_avg_dB = 10*log10( var(yAvg_clean(validIdx)) / max(var(yAvg_noise(validIdx)),1e-12) );

fprintf('\n=== SNR (global, valid region) ===\n');
fprintf('Input SNR  (Ref Mic %d): %.2f dB\n', refMicIdx, snrIn_ref_dB);
fprintf('Output SNR (Avg beam):   %.2f dB  | Gain: %.2f dB\n', snrOut_avg_dB, snrOut_avg_dB - snrIn_ref_dB);
fprintf('Output SNR (Opt beam):   %.2f dB  | Gain: %.2f dB\n', snrOut_opt_dB, snrOut_opt_dB - snrIn_ref_dB);
fprintf('Output SNR (Model BLUE): %.2f dB  | Gain: %.2f dB\n', snrOut_model_dB, snrOut_model_dB - snrIn_ref_dB);

% MSE vs ideal clean beamformer reference (consistent target)
mseAvg_ref = mean( (yAvg(validIdx) - yAvg_clean(validIdx)).^2 );
mseOpt_ref = mean( (yOpt(validIdx) - yOpt_clean(validIdx)).^2 );

fprintf('\n=== MSE vs ideal clean beamformer reference (valid region) ===\n');
fprintf('MSE (Avg vs Avg_clean):  %.6e\n', mseAvg_ref);
fprintf('MSE (Opt vs Opt_clean):  %.6e\n', mseOpt_ref);

%% ===================== 12) PLOTS: SIMULATION GEOMETRY (single, clean) =====================
figure('Name','Simulation Geometry','Position',[100 100 900 450]);

micPosPlot = micPosXY;
if all(abs(micPosPlot(2,:)) < 1e-12)
    micPosPlot(2,:) = 0.01 * sin(2*pi*(1:numMics)/numMics); % display-only jitter
end

hM = plot(micPosPlot(1,:), micPosPlot(2,:), 'ks', 'MarkerSize', 9, 'MarkerFaceColor','k'); hold on;
hS = plot(srcPosXY(1,:), srcPosXY(2,:), 'b-', 'LineWidth', 1.5);
hA = plot([-2*L 2*L], [0 0], 'k--', 'LineWidth', 0.7);
plot([0 0], [-L L], 'k--', 'LineWidth', 0.7);
hR = plot(micPosPlot(1,refMicIdx), micPosPlot(2,refMicIdx), 'ro', 'MarkerSize', 11, 'LineWidth', 2);

for i = 1:numMics
    text(micPosPlot(1,i) + 0.02, micPosPlot(2,i) + 0.01, sprintf('%d',i), ...
        'FontSize', 9, 'Color', [0.2 0.2 0.2]);
end

grid on; axis equal;
xlabel('x (m)'); ylabel('y (m)');
title('Array Geometry & Source Trajectory');

xPad = 0.4; yPad = 0.3;
xlim([min([micPosXY(1,:), srcPosXY(1,:)]) - xPad, max([micPosXY(1,:), srcPosXY(1,:)]) + xPad]);
ylim([min([micPosPlot(2,:), srcPosXY(2,:)]) - yPad, max([micPosPlot(2,:), srcPosXY(2,:)]) + yPad]);

legend([hM hS hA hR], {'Microphones','Source path','Ellipse axes','Ref Mic'}, 'Location','best');

%% ===================== 13) PLOTS: OPTIMAL WEIGHTS =====================
figure('Name','Optimal Weights Comparison','Position',[100 100 900 380]);

subplot(1,2,1);
stem(1:numMics, wOpt, 'LineWidth', 1.5); hold on;
plot(1:numMics, wAvg, '--', 'LineWidth', 1.2);
grid on;
xlabel('Microphone Index'); ylabel('Weight');
title('Empirical BLUE Weights');
legend('Optimal (empirical)','Uniform (avg)','Location','best');
xlim([0 numMics+1]);

subplot(1,2,2);
stem(1:numMics, wOpt_model, 'LineWidth', 1.5); hold on;
plot(1:numMics, wAvg, '--', 'LineWidth', 1.2);
grid on;
xlabel('Microphone Index'); ylabel('Weight');
title('Theoretical BLUE Weights');
legend('Optimal (model)','Uniform (avg)','Location','best');
xlim([0 numMics+1]);

%% ===================== 14) PLOTS: ERROR COMPARISON SUMMARY =====================
figure('Name','Error Comparison Summary','Position',[100 100 1000 400]);

subplot(1,3,1);
mseVals = [mseAvg_ref, mseOpt_ref];
bar(mseVals);
grid on;
set(gca,'XTick',1:2,'XTickLabel',{'Avg','Optimal'});
ylabel('MSE'); title('MSE vs Clean Beamformer');
yOff = 0.03*max(mseVals);
text(1, mseVals(1)+yOff, sprintf('%.2e',mseVals(1)), 'HorizontalAlignment','center');
text(2, mseVals(2)+yOff, sprintf('%.2e',mseVals(2)), 'HorizontalAlignment','center');

subplot(1,3,2);
bar(1:numMics, rmsDelayErr_ms(:).');
grid on; xlim([0 numMics+1]);
xlabel('Microphone Index'); ylabel('RMS Delay Error (ms)');
title('DToA Estimation Error');

subplot(1,3,3);
snrVals = [snrIn_ref_dB, snrOut_avg_dB, snrOut_opt_dB];
bar(1:3, snrVals);
grid on;
set(gca,'XTick',1:3,'XTickLabel',{'Ref Mic','Avg Beam','Opt Beam'});
ylabel('SNR (dB)'); title('SNR Comparison');
yOff = 0.03*max(snrVals);
for i=1:3
    text(i, snrVals(i)+yOff, sprintf('%.1f',snrVals(i)), 'HorizontalAlignment','center');
end

%% ===================== 15) PLOTS: RESIDUAL COMPARISON + WINDOWED SNR + DISPLAY =====================
tZoom1 = 2.00; tZoom2 = 2.05;
idxZ = (t >= tZoom1 & t <= tZoom2);

figure('Name','Residual Comparison (Zoom)','Position',[100 100 900 350]);
plot(t(idxZ), refNoise(idxZ), 'LineWidth', 1.0); hold on;
plot(t(idxZ), yAvg_noise(idxZ), 'LineWidth', 1.2);
plot(t(idxZ), yOpt_noise(idxZ), 'LineWidth', 1.2);
grid on;
xlabel('Time (s)'); ylabel('Amplitude');
title('Residual signals: ref mic noise vs beamformer residual');
legend('Ref mic noise','Avg residual','Opt residual','Location','best');

fprintf('\n--- Computing windowed SNR curves ---\n');
snrWin_sec = 0.20;          % 200 ms
snrHop_sec = 0.05;          % 50 ms

[snrTime, snrRef_t, snrAvg_t, snrOpt_t] = ...
    computeWindowedSnr(t, fs, validIdx, ...
                       micClean(:,refMicIdx), refNoise, ...
                       yAvg_clean, yAvg_noise, ...
                       yOpt_clean, yOpt_noise, ...
                       snrWin_sec, snrHop_sec);

figure('Name','Windowed SNR Improvement over Reference','Position',[100 100 900 350]);
plot(snrTime, snrAvg_t - snrRef_t, '--', 'LineWidth', 1.6); hold on;
plot(snrTime, snrOpt_t - snrRef_t, ':',  'LineWidth', 1.8);
grid on;
xlabel('Time (s)'); ylabel('SNR improvement (dB)');
title('Windowed SNR Improvement (Avg vs Optimal) over Reference Mic');
legend('Aligned average - ref','Optimal weights - ref','Location','best');

figure('Name','Display-only: Source vs Avg vs Optimal (zoom)','Position',[100 100 900 350]);
plot(t(idxZ), srcSig(idxZ), 'k',  'LineWidth', 1.2); hold on;
plot(t(idxZ), yAvg(idxZ),   'g--','LineWidth', 1.2);
plot(t(idxZ), yOpt(idxZ),   'r:','LineWidth', 1.3);
grid on;
xlabel('Time (s)'); ylabel('Amplitude');
title('Display-only: emitted source vs beamformer outputs (expect mismatch)');
legend('Source','Avg output','Optimal output','Location','best');
xlim([tZoom1 tZoom2]);

fprintf('\nDone.\n');

%% =============================== LOCAL FUNCTIONS ===============================

function [x, fs] = loadMonoAudio(fname, maxSec)
    if exist(fname,'file') ~= 2
        error('Audio file not found: %s', fname);
    end
    [raw, fs] = audioread(fname);
    if size(raw,2) > 1
        x = mean(raw,2);
    else
        x = raw;
    end
    nMax = min(numel(x), floor(maxSec*fs));
    x = x(1:nMax);
end

function micPosXY = buildLinearArray(N, d)
    idx = 0:(N-1);
    x = -(N-1)*d/2 + idx*d;
    micPosXY = [x; zeros(1,N)];
end

function srcPosXY = ellipticalSourceTrajectory(t, L, omega0)
    theta = omega0 * t(:)'; % 1 x Ns
    srcPosXY = [2*L*cos(theta); L*sin(theta)];
end

function [micClean, attMat] = simulatePropagation(srcSig, t, fs, srcPosXY, micPosXY, c, r0)
    Ns = numel(srcSig);
    N  = size(micPosXY,2);
    micClean = zeros(Ns, N);
    attMat   = zeros(Ns, N);

    for m = 1:N
        mp   = micPosXY(:,m);
        dist = sqrt(sum((srcPosXY - mp).^2, 1))'; % [Ns x 1]
        tau  = dist / c;

        tDelayed = t - tau;
        sDelayed = interp1(t, srcSig, tDelayed, 'linear', 0);

        distSafe = max(dist, 0.1);
        att = (r0 ./ distSafe).^2;

        micClean(:,m) = sDelayed .* att;
        attMat(:,m)   = att;
    end
end

function micNoisy = addNeighborCorrelatedNoise(micClean, snrTarget_dB, rho, seed)
    [Ns, N] = size(micClean);

    sigPow   = mean(var(micClean));                 % average across mics
    noisePow = sigPow / (10^(snrTarget_dB/10));
    sigma    = sqrt(noisePow);

    % Neighbor covariance (sensor domain)
    R = zeros(N,N);
    for i=1:N
        for j=1:N
            if i==j
                R(i,j) = sigma^2;
            elseif abs(i-j)==1
                R(i,j) = rho*sigma^2;
            end
        end
    end

    Lc = chol(R,'lower');
    rng(seed,'twister');
    u = randn(Ns, N);

    micNoisy = micClean + (u * Lc');
end

function [delayHat, delayTrue, winTime, peakQual] = estimateDelaysWindowed( ...
        micSig, t, fs, srcPosXY, micPosXY, c, refMic, winSec, overlapFrac)

    [Ns, N] = size(micSig);

    winLen = round(winSec*fs);
    hop    = max(1, round((1-overlapFrac)*winLen));
    nWin   = floor((Ns - winLen)/hop) + 1;

    delayHat  = zeros(N, nWin);
    delayTrue = zeros(N, nWin);
    peakQual  = zeros(N, nWin);
    winTime   = zeros(1, nWin);

    maxGeomDelay = ((N-1) * (micPosXY(1,2)-micPosXY(1,1)) / 2) / c;
    maxLag = round(maxGeomDelay*fs*1.5);

    w = hann(winLen);

    for kWin = 1:nWin
        a = (kWin-1)*hop + 1;
        b = a + winLen - 1;
        mid = round((a+b)/2);
        winTime(kWin) = t(mid);

        refSeg = (micSig(a:b,refMic) - mean(micSig(a:b,refMic))) .* w;

        pMid = srcPosXY(:,mid);
        dRef = norm(pMid - micPosXY(:,refMic));

        for m = 1:N
            if m == refMic
                delayHat(m,kWin)  = 0;
                delayTrue(m,kWin) = 0;
                peakQual(m,kWin)  = 1;
                continue;
            end

            x = (micSig(a:b,m) - mean(micSig(a:b,m))) .* w;
            [xc, lags] = xcorr(x, refSeg, maxLag, 'coeff');

            % Consistent peak pick + parabolic refinement 
            [pv, pi] = max(xc);
            lagSamp = lags(pi);

            if pi>1 && pi<length(xc)
                y1 = xc(pi-1); y2 = xc(pi); y3 = xc(pi+1);
                den = (y1 - 2*y2 + y3);
                if abs(den) > 1e-12
                    lagSamp = lagSamp + (y1 - y3)/(2*den);
                end
            end

            delayHat(m,kWin) = lagSamp / fs;
            peakQual(m,kWin) = pv;

            dM = norm(pMid - micPosXY(:,m));
            delayTrue(m,kWin) = (dM - dRef)/c;
        end
    end
end

function delayHat = smoothBlockDelays(delayHat, refMic, smoothLen)
    [N, nWin] = size(delayHat);
    half = floor(smoothLen/2);

    for m = 1:N
        if m == refMic, continue; end
        sm = zeros(1,nWin);
        for k = 1:nWin
            a = max(1, k-half);
            b = min(nWin, k+half);
            sm(k) = mean(delayHat(m,a:b));
        end
        delayHat(m,:) = sm;
    end
end

function rmsMs = computeRmsDelayError(delayHat, delayTrue)
    [N,~] = size(delayHat);
    rmsMs = zeros(1,N);
    for m = 1:N
        e = (delayHat(m,:) - delayTrue(m,:)) * 1000;
        rmsMs(m) = sqrt(mean(e.^2));
    end
end

function delaySample = interpolateDelaysToSamples(delayHat, winTime, t, refMic)
    N  = size(delayHat,1);
    Ns = numel(t);
    delaySample = zeros(Ns, N);

    for m = 1:N
        d = delayHat(m,:)';
        delaySample(:,m) = interp1(winTime(:), d, t, 'pchip', 'extrap');
    end
    delaySample(:,refMic) = 0;
end

function aligned = applyTimeAlignment(x, fs, delaySample)
    % aligned(:,m) = x(t + delay(t,m))
    [Ns,N] = size(x);
    aligned = zeros(Ns,N);
    t0 = (0:Ns-1)'/fs;

    for m = 1:N
        tShift = t0 + delaySample(:,m);
        aligned(:,m) = interp1(t0, x(:,m), tShift, 'linear', 0);
    end
end

function Rn = neighborCovariance(N, sigma2, rho)
    Rn = zeros(N,N);
    for i=1:N
        for j=1:N
            if i==j
                Rn(i,j) = sigma2;
            elseif abs(i-j)==1
                Rn(i,j) = rho*sigma2;
            end
        end
    end
end

function [snrTime, snrRef, snrAvg, snrOpt] = computeWindowedSnr( ...
    t, fs, validIdx, refClean, refNoise, avgClean, avgNoise, optClean, optNoise, winSec, hopSec)

    win = max(128, round(winSec*fs));
    hop = max(1, round(hopSec*fs));

    starts = validIdx(1):hop:(validIdx(end)-win+1);
    K = numel(starts);

    snrTime = zeros(K,1);
    snrRef  = zeros(K,1);
    snrAvg  = zeros(K,1);
    snrOpt  = zeros(K,1);

    for k = 1:K
        a = starts(k);
        b = a + win - 1;
        snrTime(k) = t(a + floor(win/2));

        snrRef(k) = 10*log10( var(refClean(a:b)) / max(var(refNoise(a:b)),1e-12) );
        snrAvg(k) = 10*log10( var(avgClean(a:b)) / max(var(avgNoise(a:b)),1e-12) );
        snrOpt(k) = 10*log10( var(optClean(a:b)) / max(var(optNoise(a:b)),1e-12) );
    end
end
