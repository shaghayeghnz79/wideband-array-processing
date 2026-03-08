%% Part (b) 


clearvars -except fs Ns t s s1_clean s2_clean g1 g2 frame_centers Lw maxLag upsamp

%% -------------------- 0) Sanity checks --------------------
req = {'fs','Ns','t','s','s1_clean','s2_clean','g1','g2','frame_centers','Lw','maxLag','upsamp'};
for i = 1:numel(req)
    if ~exist(req{i},'var')
        error('Missing required variable "%s". Run Part (a) first.', req{i});
    end
end
if numel(frame_centers) < 2
    error('frame_centers must contain at least 2 entries (need hop).');
end

%% -------------------- 1) User settings --------------------
SNRref_dB_list     = 0:5:30;   % MSE curves vs SNRref
MC                = 40;        % Monte Carlo runs per SNR
smooth_sec        = 0.5;       % smoothing time for Δτ̂ (seconds)
doListenExample   = true;      % export example WAVs
SNRref_listen_dB  = 20;        % SNR for WAV export
doAddonPlots      = true;      % plots (6)(7)(8) using the listening example

%% -------------------- 2) Time axes and smoothing window --------------------
Ps = var(s);                         % source power (~1 if normalized)
tt = (0:Ns-1)'/fs;
t_frames = t(frame_centers);

hop_samp = frame_centers(2) - frame_centers(1);
hop_sec  = hop_samp/fs;
smooth_win_frames = max(3, round(smooth_sec / hop_sec));

fprintf('Part (b): smoothing Δτ̂ with movmean over %.2fs -> %d frames.\n', ...
    smooth_sec, smooth_win_frames);

% Helper to build sample-wise dtau_hat(t) from noisy signals
build_dtau_hat_samplewise = @(r1_noisy, r2_noisy) ...
    dtau_hat_samplewise_from_frames( ...
        estimate_dtau_frames_xcorr(r1_noisy, r2_noisy, fs, frame_centers, Lw, maxLag, upsamp), ...
        t_frames, tt, smooth_win_frames);

%% -------------------- 3) Monte Carlo: MSE1 and MSE2 vs SNRref --------------------
MSE1 = zeros(numel(SNRref_dB_list),1);
MSE2 = zeros(numel(SNRref_dB_list),1);

s_ref = s;   % reference for comparison

valid = (floor(0.5*fs)+1) : (Ns-floor(0.2*fs));   % avoid edge interpolation regions
maxLag_bulk = round(0.05*fs);                     % 50 ms bulk lag search

for k = 1:numel(SNRref_dB_list)
    SNRref_dB  = SNRref_dB_list(k);
    SNRref_lin = 10^(SNRref_dB/10);
    sigma = sqrt(Ps / SNRref_lin);    % global SNR model

    mse1_acc = 0;
    mse2_acc = 0;

    for mc = 1:MC
        % Noisy received signals
        r1_noisy = s1_clean + sigma*randn(Ns,1);
        r2_noisy = s2_clean + sigma*randn(Ns,1);

        % Sample-wise dtau_hat(t)
        dtau_hat_s = build_dtau_hat_samplewise(r1_noisy, r2_noisy);

        % Symmetric realignment
        alpha1 = -0.5 * dtau_hat_s;
        alpha2 = +0.5 * dtau_hat_s;

        R1_aligned = timevarying_timeshift(r1_noisy, tt, alpha1, 'pchip');
        R2_aligned = timevarying_timeshift(r2_noisy, tt, alpha2, 'pchip');

        % Amplitude correction
        R1_corr = R1_aligned ./ (sqrt(g1) + 1e-6);
        R2_corr = R2_aligned ./ (sqrt(g2) + 1e-6);

        % Remove constant residual lag vs reference
        lag1 = bulk_lag_xcorr(R1_corr, s_ref, maxLag_bulk);
        lag2 = bulk_lag_xcorr(R2_corr, s_ref, maxLag_bulk);
        R1_corr = align_by_integer_lag(R1_corr, lag1);
        R2_corr = align_by_integer_lag(R2_corr, lag2);

        % MSE vs s(t)
        e1 = R1_corr(valid) - s_ref(valid);
        e2 = R2_corr(valid) - s_ref(valid);
        mse1_acc = mse1_acc + mean(e1.^2);
        mse2_acc = mse2_acc + mean(e2.^2);
    end

    MSE1(k) = mse1_acc/MC;
    MSE2(k) = mse2_acc/MC;

    fprintf('SNRref=%3d dB:  MSE1=%.3e, MSE2=%.3e\n', SNRref_dB, MSE1(k), MSE2(k));
end

%% -------------------- 4) Plot: MSE1, MSE2 vs SNRref --------------------
figure('Name','Part (b) Reconstruction MSE vs SNRref'); hold on; grid on;
semilogy(SNRref_dB_list, MSE1, 'o-','LineWidth',1.5);
semilogy(SNRref_dB_list, MSE2, 'o-','LineWidth',1.5);
xlabel('SNR_{ref} (dB)');
ylabel('MSE (aligned + amp-corrected vs s(t))');
title('Part (b) Reconstruction MSE vs SNR_{ref}');
legend('MSE_1 (Mic 1)','MSE_2 (Mic 2)','Location','northeast');
set(gca,'YScale','log');

%% -------------------- 5) one listening example --------------------
if doListenExample || doAddonPlots
    SNRref_lin = 10^(SNRref_listen_dB/10);
    sigma = sqrt(Ps / SNRref_lin);

    r1_noisy = s1_clean + sigma*randn(Ns,1);
    r2_noisy = s2_clean + sigma*randn(Ns,1);

    dtau_hat_s = build_dtau_hat_samplewise(r1_noisy, r2_noisy);
    alpha1 = -0.5 * dtau_hat_s;
    alpha2 = +0.5 * dtau_hat_s;

    R1_aligned = timevarying_timeshift(r1_noisy, tt, alpha1, 'pchip');
    R2_aligned = timevarying_timeshift(r2_noisy, tt, alpha2, 'pchip');

    R1_corr = R1_aligned ./ (sqrt(g1) + 1e-6);
    R2_corr = R2_aligned ./ (sqrt(g2) + 1e-6);

    if doListenExample
        rec_stereo = normalize_audio([r1_noisy, r2_noisy]);
        out_stereo = normalize_audio([R1_corr, R2_corr]);

        audiowrite(sprintf('received_SNR%ddB.wav', SNRref_listen_dB), rec_stereo, fs);
        audiowrite(sprintf('reconstructed_SNR%ddB.wav', SNRref_listen_dB), out_stereo, fs);

        fprintf('WAV exported: received_SNR%ddB.wav and reconstructed_SNR%ddB.wav\n', ...
            SNRref_listen_dB, SNRref_listen_dB);
    end
end

%% -------------------- 6–8) APlotting: Before/After DToA, Robust zoom, Spectrogram --------------------
if doAddonPlots

    R1_after = R1_aligned;
    R2_after = R2_aligned;


    % (6) DToA estimate before vs after
    dtau_before_frames = estimate_dtau_frames_xcorr(r1_noisy, r2_noisy, fs, frame_centers, Lw, maxLag, upsamp);
    dtau_after_frames  = estimate_dtau_frames_xcorr(R1_after, R2_after, fs, frame_centers, Lw, maxLag, upsamp);

    figure('Name','(6) DToA before vs after realignment'); hold on; grid on;
    plot(t_frames, dtau_before_frames, 'LineWidth', 1.3);
    plot(t_frames, dtau_after_frames,  '--', 'LineWidth', 1.3);
    yline(0, ':', 'LineWidth', 1.1);
    xlabel('Time (s)'); ylabel('\Delta\tau estimate (s)');
    title(sprintf('DToA estimate before vs after (SNR_{ref}=%d dB)', SNRref_listen_dB));
    legend('\Delta\tau_{before}','\Delta\tau_{after}','0 ref','Location','best');
    fprintf('[Plot 6] Std before=%.3e s, Std after=%.3e s\n', std(dtau_before_frames), std(dtau_after_frames));

    % (7) robust zoom: show inter-channel consistency improvement (NOT s(t) overlay)
    plot7_alignment_zoom(r1_noisy, r2_noisy, R1_after, R2_after, tt, fs, Ns, frame_centers, dtau_before_frames);

  
end

%% ========================= LOCAL FUNCTIONS =========================
function dtau_samp = dtau_hat_samplewise_from_frames(dtau_frames, t_frames, t_samp, smooth_win_frames)
% Smooth frame-wise dtau and interpolate to sample time axis.
    dtau_smooth = smoothdata(dtau_frames(:), 'movmean', smooth_win_frames);
    dtau_samp = interp1(t_frames(:), dtau_smooth, t_samp(:), 'pchip', 'extrap');
end

function y = timevarying_timeshift(x, t, alpha, method)
% y(t) = x(t + alpha(t)) via interpolation (time-varying resampling).
    tq = t + alpha(:);
    y = interp1(t, x, tq, method, 0);   % out-of-range -> 0
end

function lag = bulk_lag_xcorr(x, ref, maxLag)
% Find integer lag that maximizes |xcorr| between x and ref.
    [c,lags] = xcorr(x, ref, maxLag, 'coeff');
    [~,k] = max(abs(c));
    lag = lags(k);
end

function y = align_by_integer_lag(x, lag)
% Shift signal by integer lag (samples).
    N = numel(x);
    y = zeros(N,1);
    if lag > 0
        y(1:N-lag) = x(1+lag:N);
    elseif lag < 0
        L = -lag;
        y(1+L:N) = x(1:N-L);
    else
        y = x;
    end
end

function y = normalize_audio(x)
% Normalize to avoid clipping.
    m = max(abs(x(:)));
    if m < 1e-12, y = x; return; end
    y = 0.99 * (x / m);
end

function dtau_hat = estimate_dtau_frames_xcorr(r1, r2, fs, centers, Lw, maxLag, upsamp)
% Frame-wise wideband DToA estimator using normalized xcorr + local upsample.
    B = numel(centers);
    dtau_hat = zeros(B,1);
    w = hann(Lw);

    for b = 1:B
        c0  = centers(b);
        idx = (c0-floor(Lw/2)):(c0+ceil(Lw/2)-1);

        f1 = r1(idx).*w;
        f2 = r2(idx).*w;

        [c,lags] = xcorr(f2, f1, maxLag, 'coeff');
        [~,k0] = max(abs(c));

        win = 6;
        kL = max(1, k0-win);
        kR = min(numel(c), k0+win);

        l_f = linspace(lags(kL), lags(kR), (kR-kL+1)*upsamp);
        c_f = interp1(lags(kL:kR), c(kL:kR), l_f, 'pchip');

        [~,kf] = max(abs(c_f));
        dtau_hat(b) = l_f(kf)/fs;
    end
end

function plot7_alignment_zoom(r1, r2, R1, R2, tt, fs, Ns, frame_centers, dtau_before_frames)
% Robust zoom plot (Part b): shows improvement in inter-channel consistency.
    if nargin < 9 || isempty(dtau_before_frames)
        n0 = round(Ns/2);
    else
        [~, bMax] = max(abs(dtau_before_frames));
        n0 = frame_centers(bMax);
    end

    win_ms   = 80;
    win_samp = round((win_ms/1000)*fs);
    idx = max(1, n0-floor(win_samp/2)) : min(Ns, n0+ceil(win_samp/2)-1);
    tzoom = tt(idx);

    b1 = r1(idx); b2 = r2(idx);
    a1 = R1(idx); a2 = R2(idx);

    scale = max([abs(b1); abs(b2); abs(a1); abs(a2)]) + eps;

    e_before = (b2 - b1)/scale;
    e_after  = (a2 - a1)/scale;

    rms_before = sqrt(mean((b2 - b1).^2));
    rms_after  = sqrt(mean((a2 - a1).^2));
    impr_dB = 20*log10(rms_before / max(rms_after,eps));

    figure('Name','(7) Alignment zoom: before vs after (robust)');
    tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

    nexttile; hold on; grid on;
    plot(tzoom, b1/scale, '--', 'LineWidth', 1.1);
    plot(tzoom, b2/scale, '--', 'LineWidth', 1.1);
    plot(tzoom, a1/scale, '-',  'LineWidth', 1.3);
    plot(tzoom, a2/scale, '-',  'LineWidth', 1.3);
    xlabel('Time (s)'); ylabel('Normalized amplitude');
    title(sprintf('Zoom (%.0f ms): channels before (dashed) vs after (solid)', win_ms));
    legend('Mic1 before','Mic2 before','Mic1 after','Mic2 after','Location','best');

    nexttile; hold on; grid on;
    plot(tzoom, e_before, '--', 'LineWidth', 1.2);
    plot(tzoom, e_after,  '-',  'LineWidth', 1.4);
    yline(0, ':');
    xlabel('Time (s)'); ylabel('(Ch2 - Ch1) / scale');
    title(sprintf('Residual mismatch: RMS before=%.3e, after=%.3e (%.1f dB improvement)', ...
        rms_before, rms_after, impr_dB));
    legend('Before: r2-r1','After: R2-R1','0','Location','best');

    fprintf('[Plot 7] Residual (ch2-ch1) RMS: before=%.3e, after=%.3e => improvement=%.1f dB\n', ...
        rms_before, rms_after, impr_dB);
end
